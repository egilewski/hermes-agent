#!/usr/bin/env node

import { spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  appendFileSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs'
import http from 'node:http'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { CDP, discoverTarget, sleep } from './perf/lib/cdp.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const DESKTOP_ROOT = resolve(HERE, '..')
const REPO_ROOT = resolve(DESKTOP_ROOT, '..', '..')
const HARNESS_SOURCE = resolve(DESKTOP_ROOT, 'src/app/chat/short-session-hang-repro.tsx')
const UPSTREAM_URL = 'https://github.com/NousResearch/hermes-agent.git'
const DEFAULT_BASELINE = 'c6f9e0c748677fcc46f62b71cc99a9069239de5b'
const DEFAULT_CANDIDATE = 'refs/pull/69275/head'
const FREEZE_MS = 5_000
const OUTER_WATCHDOG_MS = 120_000
const SECRET_ENV_RE = /(credential|token|secret|password|(^|_)key($|_)|auth|cookie)/i
const HEX_OBJECT_RE = /^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$/
const PULL_REF_RE = /^refs\/pull\/[1-9][0-9]*\/(?:head|merge)$/
const HEAD_REF_RE = /^refs\/heads\/[A-Za-z0-9][A-Za-z0-9._/-]*$/
const HEAD_COMPONENT_RE = /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/
const require = createRequire(import.meta.url)

class ReproductionError extends Error {
  name = 'ReproductionError'
}

function usage() {
  console.log(`Usage: node scripts/run-short-session-hang-repro.mjs [options]

Options:
  --baseline <ref>       baseline ref (default: ${DEFAULT_BASELINE})
  --candidate <ref>      candidate ref (default: ${DEFAULT_CANDIDATE})
  --repetitions <n>      measured repetitions per ref (default: 5)
  --output <dir>         artifact directory (default: short-session-hang-artifacts)
  --keep-worktrees       retain ephemeral source copies
  --dry-run              validate refs, lockfile/Electron parity, and print the plan
  --help                 show this help

Each ref gets one warm-up plus N measured fresh-app runs. Measured A/B order is
counterbalanced. A run is a reproduction only when a renderer/main operation,
heartbeat, or event-loop gap exceeds ${FREEZE_MS}ms, or Electron becomes
unresponsive, loses the renderer, or exits unexpectedly.`)
}

function parseArgs(argv) {
  const out = {
    baseline: DEFAULT_BASELINE,
    candidate: DEFAULT_CANDIDATE,
    repetitions: 5,
    output: resolve(process.cwd(), 'short-session-hang-artifacts'),
    dryRun: false,
    keepWorktrees: false
  }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    const next = argv[i + 1]

    if (arg === '--help') {
      usage()
      process.exit(0)
    } else if (arg === '--dry-run') {
      out.dryRun = true
    } else if (arg === '--keep-worktrees') {
      out.keepWorktrees = true
    } else if (arg === '--baseline' || arg === '--candidate' || arg === '--repetitions' || arg === '--output') {
      if (!next || next.startsWith('--')) {
        throw new Error(`${arg} requires a value`)
      }

      const key = arg.slice(2)
      out[key] = key === 'repetitions' ? Number(next) : key === 'output' ? resolve(next) : next
      i += 1
    } else {
      throw new Error(`unknown option: ${arg}`)
    }
  }

  if (!Number.isInteger(out.repetitions) || out.repetitions < 1 || out.repetitions > 20) {
    throw new Error('--repetitions must be an integer from 1 to 20')
  }

  return out
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? REPO_ROOT,
    encoding: 'utf8',
    env: options.env ?? process.env,
    maxBuffer: 64 * 1024 * 1024
  })

  if (options.logPath) {
    writeFileSync(options.logPath, `${result.stdout ?? ''}${result.stderr ?? ''}`)
  }

  if (result.error) {
    throw new Error(`${command} ${args.join(' ')} failed to run: ${result.error.message}`)
  }

  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.join(' ')} exited ${result.status ?? `signal ${result.signal}`}:\n${result.stderr || result.stdout}`
    )
  }

  return String(result.stdout ?? '').trim()
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function validateRef(ref) {
  const headParts = ref.startsWith('refs/heads/') ? ref.slice('refs/heads/'.length).split('/') : []
  const invalidHead =
    ref.includes('..') ||
    headParts.length === 0 ||
    headParts.some(part => !HEAD_COMPONENT_RE.test(part) || part.endsWith('.lock'))

  if (HEX_OBJECT_RE.test(ref) || PULL_REF_RE.test(ref) || (HEAD_REF_RE.test(ref) && !invalidHead)) {
    return ref
  }

  throw new Error(
    `unsafe ref ${JSON.stringify(ref)}; use a 40/64-digit hex object ID, refs/heads/<safe>, or refs/pull/<number>/(head|merge)`
  )
}

function sanitizedEnv(extra = {}) {
  const clean = Object.fromEntries(Object.entries(process.env).filter(([name]) => !SECRET_ENV_RE.test(name)))

  return { ...clean, ...extra }
}

function resolveRef(rawRef) {
  const ref = validateRef(rawRef)
  const tryResolve = () => {
    const result = spawnSync('git', ['rev-parse', '--verify', '--end-of-options', `${ref}^{commit}`], {
      cwd: REPO_ROOT,
      encoding: 'utf8'
    })

    return result.status === 0 ? result.stdout.trim() : null
  }

  const local = HEX_OBJECT_RE.test(ref) ? tryResolve() : null

  if (local) {
    return local
  }

  run('git', ['fetch', '--no-tags', UPSTREAM_URL, ref], {
    env: sanitizedEnv({ GCM_INTERACTIVE: 'never', GIT_TERMINAL_PROMPT: '0' })
  })

  if (ref.startsWith('refs/')) {
    const fetched = run('git', ['rev-parse', '--verify', '--end-of-options', 'FETCH_HEAD^{commit}'])

    return fetched
  }

  const resolved = tryResolve()

  if (!resolved) {
    throw new Error(`cannot resolve ref ${ref}`)
  }

  return resolved
}

function readTargetMetadata(sha) {
  const lock = run('git', ['show', '--end-of-options', `${sha}:package-lock.json`])
  const pyproject = run('git', ['show', '--end-of-options', `${sha}:pyproject.toml`])
  const uvLock = run('git', ['show', '--end-of-options', `${sha}:uv.lock`])
  const pkg = JSON.parse(run('git', ['show', '--end-of-options', `${sha}:apps/desktop/package.json`]))

  return {
    electron: pkg.devDependencies?.electron,
    electronBuild: pkg.build?.electronVersion,
    lockSha256: sha256(lock),
    pyprojectSha256: sha256(pyproject),
    uvLockSha256: sha256(uvLock)
  }
}

function injectHarness(targetRoot) {
  const targetHarness = join(targetRoot, 'apps/desktop/src/app/chat/short-session-hang-repro.tsx')
  const targetMain = join(targetRoot, 'apps/desktop/src/main.tsx')
  const main = readFileSync(targetMain, 'utf8')

  mkdirSync(dirname(targetHarness), { recursive: true })
  copyFileSync(HARNESS_SOURCE, targetHarness)

  if (!main.includes("import('./app/chat/short-session-hang-repro')")) {
    writeFileSync(
      targetMain,
      `${main}\nif (import.meta.env.VITE_SHORT_SESSION_HANG_REPRO === '1') {\n  import('./app/chat/short-session-hang-repro')\n}\n`
    )
  }
}

function linkShared(targetRoot, relativePath) {
  const source = join(REPO_ROOT, relativePath)
  const target = join(targetRoot, relativePath)

  if (existsSync(source) && !existsSync(target)) {
    mkdirSync(dirname(target), { recursive: true })
    symlinkSync(source, target, 'dir')
  }
}

function prepareTarget(label, sha, root, output) {
  const targetRoot = join(root, label)
  run('git', ['worktree', 'add', '--detach', targetRoot, sha])

  try {
    injectHarness(targetRoot)
    linkShared(targetRoot, 'node_modules')
    linkShared(targetRoot, 'apps/desktop/node_modules')
    linkShared(targetRoot, '.venv')

    const targetDesktop = join(targetRoot, 'apps/desktop')
    const buildLog = join(output, `${label}-build.log`)
    run('npm', ['run', '--prefix', 'apps/desktop', 'build'], {
      cwd: targetRoot,
      env: sanitizedEnv({ VITE_SHORT_SESSION_HANG_REPRO: '1' }),
      logPath: buildLog
    })

    return { label, sha, targetDesktop, targetRoot }
  } catch (error) {
    try {
      run('git', ['worktree', 'remove', '--force', targetRoot])
    } catch {
      // The original build error is more useful; cleanup is retried manually.
    }

    throw error
  }
}

function startMockInference() {
  const reply = 'Deterministic local short-session diagnostic response.'
  let streamingCompletionRequests = 0
  const server = http.createServer((request, response) => {
    if (request.method === 'GET' && request.url === '/v1/models') {
      response.writeHead(200, { 'content-type': 'application/json' })
      response.end(JSON.stringify({ data: [{ id: 'short-session-model', object: 'model' }], object: 'list' }))

      return
    }

    if (request.method === 'POST' && request.url?.startsWith('/v1/chat/completions')) {
      let body = ''
      request.on('data', chunk => {
        body += String(chunk)
      })
      request.on('end', () => {
        let stream = false

        try {
          stream = JSON.parse(body).stream === true
        } catch {
          stream = false
        }

        if (stream) {
          streamingCompletionRequests += 1
          response.writeHead(200, { 'content-type': 'text/event-stream' })
          response.write(
            `data: ${JSON.stringify({ choices: [{ delta: { content: reply }, finish_reason: null, index: 0 }], id: 'short-session', object: 'chat.completion.chunk' })}\n\n`
          )
          response.write(
            `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop', index: 0 }], id: 'short-session', object: 'chat.completion.chunk' })}\n\n`
          )
          response.end('data: [DONE]\n\n')
        } else {
          response.writeHead(200, { 'content-type': 'application/json' })
          response.end(
            JSON.stringify({
              choices: [{ finish_reason: 'stop', index: 0, message: { content: reply, role: 'assistant' } }],
              id: 'short-session',
              object: 'chat.completion'
            })
          )
        }
      })

      return
    }

    response.writeHead(404, { 'content-type': 'application/json' })
    response.end('{"error":"not found"}')
  })

  return new Promise((resolveStart, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()

      if (!address || typeof address === 'string') {
        reject(new Error('mock inference server has no TCP address'))

        return
      }

      resolveStart({
        url: `http://127.0.0.1:${address.port}`,
        streamingCompletionRequests: () => streamingCompletionRequests,
        close: () => new Promise(resolveClose => server.close(() => resolveClose()))
      })
    })
  })
}

function writeSandboxConfig(home, mockUrl) {
  mkdirSync(home, { recursive: true })
  writeFileSync(
    join(home, 'config.yaml'),
    `model:\n  default: short-session-model\n  provider: short-session\nauxiliary:\n  title_generation:\n    enabled: false\nproviders:\n  short-session:\n    api: ${mockUrl}/v1\n    api_mode: chat_completions\n    key_env: SHORT_SESSION_API_KEY\n    models:\n      short-session-model: {}\n`
  )
  writeFileSync(join(home, '.env'), 'SHORT_SESSION_API_KEY=local-diagnostic-only\n')
}

async function waitFor(cdp, expression, timeoutMs, label, ErrorType = Error) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    try {
      if (await cdp.eval(expression)) {
        return
      }
    } catch {
      // Renderer is still loading.
    }

    await sleep(Math.min(250, Math.max(1, deadline - Date.now())))
  }

  throw new ErrorType(`timed out waiting for ${label}`)
}

async function screenshot(cdp, path) {
  try {
    await cdp.send('Page.enable')
    const shot = await cdp.send('Page.captureScreenshot', { format: 'png', fromSurface: true })
    writeFileSync(path, Buffer.from(shot.data, 'base64'))
  } catch {
    // A frozen renderer may not service screenshot capture.
  }
}

function reliablePid(pid) {
  return Number.isSafeInteger(pid) && pid > 1 ? pid : null
}

function processRows() {
  const result = spawnSync('ps', ['-axo', 'pid=,ppid=,%cpu=,%mem=,state=,etime=,command='], { encoding: 'utf8' })

  return String(result.stdout ?? '')
    .split(/\r?\n/)
    .map(line => {
      const match = line.match(/^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(.*)$/)

      return match
        ? {
            command: match[7],
            cpu: Number(match[3]),
            elapsed: match[6],
            memory: Number(match[4]),
            pid: Number(match[1]),
            ppid: Number(match[2]),
            state: match[5]
          }
        : null
    })
    .filter(Boolean)
}

function processTree(rootPid) {
  if (!reliablePid(rootPid)) {
    return []
  }

  const rows = processRows()
  const selected = new Set([rootPid])
  let changed = true

  while (changed) {
    changed = false

    for (const row of rows) {
      if (!selected.has(row.pid) && selected.has(row.ppid)) {
        selected.add(row.pid)
        changed = true
      }
    }
  }

  return rows.filter(row => selected.has(row.pid))
}

function redactCommand(command) {
  return command
    .replace(/([a-z][a-z0-9+.-]*:\/\/)[^\s/@]+(?::[^\s/@]*)?@/gi, '$1[REDACTED]@')
    .replace(/\b([A-Za-z_][A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|KEY|AUTH|COOKIE)[A-Za-z0-9_]*)=\S+/gi, '$1=[REDACTED]')
    .replace(/(--?\S*(?:token|secret|password|key|auth|cookie)\S*)(?:=|\s+)\S+/gi, '$1=[REDACTED]')
}

function processSnapshot(path, rootPid) {
  const rows = processTree(rootPid).map(row => ({ ...row, command: redactCommand(row.command) }))
  writeFileSync(path, `${JSON.stringify({ rootPid, rows }, null, 2)}\n`)

  return rows
}

function sampleOne(path, pid) {
  if (process.platform !== 'darwin' || !reliablePid(pid)) {
    return
  }

  const result = spawnSync('sample', [String(pid), '5', '1'], { encoding: 'utf8', timeout: 8_000 })
  writeFileSync(path, `${result.stdout ?? ''}${result.stderr ?? ''}`)
}

function sampleTree(runDir, rootPid, prefix) {
  const rows = processTree(rootPid)
  sampleOne(join(runDir, `${prefix}-main.sample.txt`), rootPid)
  const renderer = rows
    .filter(row => row.pid !== rootPid && /(?:^|\s)--type=renderer(?:\s|$)/.test(row.command))
    .sort((a, b) => b.cpu - a.cpu)[0]

  if (renderer) {
    sampleOne(join(runDir, `${prefix}-renderer-${renderer.pid}.sample.txt`), renderer.pid)
  }
}

function liveCaptured(captured) {
  const current = new Map(processRows().map(row => [row.pid, row]))

  return captured.filter(original => {
    const row = current.get(original.pid)

    return row && !row.state.startsWith('Z') && row.command === original.command
  })
}

async function stopProcessTree(rootPid) {
  const captured = processTree(rootPid)

  for (const { pid } of [...captured].reverse()) {
    try {
      process.kill(pid, 'SIGTERM')
    } catch {
      // It already exited.
    }
  }

  const deadline = Date.now() + 3_000

  while (Date.now() < deadline && liveCaptured(captured).length > 0) {
    await sleep(100)
  }

  const remaining = liveCaptured(captured)

  for (const { pid } of remaining.reverse()) {
    try {
      process.kill(pid, 'SIGKILL')
    } catch {
      // It exited between the liveness check and signal.
    }
  }

  const killDeadline = Date.now() + 2_000

  while (Date.now() < killDeadline && liveCaptured(captured).length > 0) {
    await sleep(100)
  }

  return {
    captured: captured.map(row => row.pid),
    remainingAfterKill: liveCaptured(captured).map(row => row.pid)
  }
}

function withWatchdog(task, onTimeout) {
  let timer

  return Promise.race([
    task,
    new Promise((_, reject) => {
      timer = setTimeout(() => {
        onTimeout()
        reject(new ReproductionError(`outer watchdog exceeded ${OUTER_WATCHDOG_MS}ms`))
      }, OUTER_WATCHDOG_MS)
    })
  ]).finally(() => clearTimeout(timer))
}

function withTimeout(task, timeoutMs, label, ErrorType = Error) {
  let timer

  return Promise.race([
    task,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new ErrorType(`${label} exceeded ${timeoutMs}ms`)), timeoutMs)
    })
  ]).finally(() => clearTimeout(timer))
}

async function waitForResponsive(cdp, expression, timeoutMs, label, evaluationTimeoutMs = FREEZE_MS) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    try {
      if (
        await withTimeout(cdp.eval(expression), evaluationTimeoutMs, `${label} renderer evaluation`, ReproductionError)
      ) {
        return
      }
    } catch (error) {
      if (error instanceof ReproductionError) {
        throw error
      }

      // Match the initial renderer-readiness polling: CDP can fail transiently while a document is replaced.
    }

    await sleep(Math.min(250, Math.max(1, deadline - Date.now())))
  }

  throw new Error(`timed out waiting for ${label} while the renderer remained responsive`)
}

async function waitForPredicate(predicate, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (await predicate()) {
      return
    }

    await sleep(Math.min(100, Math.max(1, deadline - Date.now())))
  }

  throw new Error(`timed out waiting for ${label}`)
}

async function verifyInteractiveSurfaces(cdp, timed, measure, label) {
  const sentinel = `short-session-sentinel-${label}`
  const composerPainted = await timed(`composer.paint.${label}`, async () => {
    const focused = await cdp.eval(
      `(() => { const el = document.querySelector('[data-slot="composer-rich-input"]'); if (!el || el.contentEditable !== 'true') return false; el.focus(); return true })()`
    )

    if (!focused) {
      return false
    }

    await cdp.send('Input.insertText', { text: sentinel })

    return cdp.eval(
      `document.querySelector('[data-slot="composer-rich-input"]')?.textContent?.includes(${JSON.stringify(sentinel)}) === true`
    )
  })

  if (!composerPainted) {
    throw new Error(`composer did not paint sentinel at ${label}`)
  }

  const version = await timed(`version.ipc.${label}`, () => cdp.eval('window.hermesDesktop.getVersion()'))
  await timed(`about.open.${label}`, () => cdp.eval("location.hash = '#/settings?tab=about'; true"))
  await measure(`about.ready.${label}`, () =>
    waitForResponsive(
      cdp,
      `document.body.textContent.includes(${JSON.stringify(version.appVersion)}) && !!document.querySelector('button[aria-label]')`,
      FREEZE_MS,
      `About settings at ${label}`
    )
  )
  const aboutClosed = await timed(`about.close.${label}`, () =>
    cdp.eval(
      `(() => { const close = document.querySelector('div[role="presentation"] > div > div:first-child button[aria-label]'); if (!close) return false; close.click(); return true })()`
    )
  )

  if (!aboutClosed) {
    throw new Error(`About settings close control unavailable at ${label}`)
  }

  await measure(`about.closed.${label}`, () =>
    waitForResponsive(cdp, "!location.hash.includes('/settings')", FREEZE_MS, `About settings close at ${label}`)
  )
  const interactive = await timed(`transcript.interactive.${label}`, () =>
    cdp.eval(
      `(() => { const viewport = document.querySelector('[data-slot="aui_thread-viewport"]'); const row = document.querySelector('[data-message-id]'); if (!viewport || !row) return false; const before = viewport.scrollTop; viewport.scrollTop = Math.min(viewport.scrollHeight, before + 40); viewport.dispatchEvent(new Event('scroll', { bubbles: true })); return getComputedStyle(row).pointerEvents !== 'none' })()`
    )
  )

  if (!interactive) {
    throw new Error(`transcript was not interactive at ${label}`)
  }

  return { sentinel, version }
}

async function runRealChatChecks(cdp, timed, measure, mock, runDir) {
  const requestCountBefore = mock.streamingCompletionRequests()

  for (let exchange = 1; exchange <= 5; exchange += 1) {
    const beforeAssistant = await timed(`real-chat.assistant-count.${exchange}`, () =>
      cdp.eval(
        `document.querySelectorAll('[data-slot="aui_assistant-message-root"]:not([data-streaming="true"])').length`
      )
    )
    const composer = await timed(`real-chat.composer-focus.${exchange}`, () =>
      cdp.eval(
        `(() => { const el = document.querySelector('[data-slot="composer-rich-input"]'); if (!el || el.contentEditable !== 'true') return false; el.focus(); return true })()`
      )
    )

    if (!composer) {
      throw new Error(`real chat composer unavailable at exchange ${exchange}`)
    }

    const prompt = `Deterministic real chat exchange ${exchange}`
    await timed(`real-chat.insert.${exchange}`, () => cdp.send('Input.insertText', { text: prompt }))
    const inserted = await timed(`real-chat.inserted.${exchange}`, () =>
      cdp.eval(
        `document.querySelector('[data-slot="composer-rich-input"]')?.textContent?.includes(${JSON.stringify(prompt)}) === true`
      )
    )

    if (!inserted) {
      throw new Error(`real chat prompt did not reach the composer at exchange ${exchange}`)
    }

    const enter = { code: 'Enter', key: 'Enter', windowsVirtualKeyCode: 13 }
    await timed(`real-chat.enter-down.${exchange}`, () =>
      cdp.send('Input.dispatchKeyEvent', { ...enter, text: '\r', type: 'keyDown', unmodifiedText: '\r' })
    )
    await timed(`real-chat.enter-up.${exchange}`, () => cdp.send('Input.dispatchKeyEvent', { ...enter, type: 'keyUp' }))

    try {
      await measure(`real-chat.mock-request.${exchange}`, () =>
        waitForPredicate(
          () => mock.streamingCompletionRequests() >= requestCountBefore + exchange,
          FREEZE_MS,
          `mock inference request ${exchange}`
        )
      )
    } catch (error) {
      const state = await timed(`real-chat.dispatch-probe.${exchange}`, () =>
        cdp.eval(`({
          assistantMessages: document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length,
          composerText: document.querySelector('[data-slot="composer-rich-input"]')?.textContent ?? null,
          harness: window.__SHORT_SESSION_HANG_REPRO__.summary(),
          userMessages: document.querySelectorAll('[data-slot="aui_user-message-root"]').length
        })`)
      )

      throw new Error(
        `${error instanceof Error ? error.message : String(error)}; dispatch probe responded with state: ${JSON.stringify(state)}`
      )
    }

    await measure(`real-chat.assistant-response.${exchange}`, () =>
      waitForResponsive(
        cdp,
        `document.querySelectorAll('[data-slot="aui_assistant-message-root"]:not([data-streaming="true"])').length > ${beforeAssistant}`,
        FREEZE_MS,
        `real assistant response ${exchange}`
      )
    )

    if (mock.streamingCompletionRequests() < requestCountBefore + exchange) {
      throw new Error(`mock inference request count did not advance for exchange ${exchange}`)
    }
  }

  const requestDelta = mock.streamingCompletionRequests() - requestCountBefore

  if (requestDelta !== 5) {
    throw new Error(`expected exactly 5 mock completion requests, observed ${requestDelta}`)
  }

  const surfaces = await verifyInteractiveSurfaces(cdp, timed, measure, 'real-chat-exchange-5')
  await withTimeout(screenshot(cdp, join(runDir, 'real-chat-exchange-5.png')), 2_000, 'real chat screenshot')

  return {
    assistantResponses: 5,
    exchanges: 5,
    messageRecords: 10,
    mockCompletionRequests: requestDelta,
    surfaces
  }
}

async function runRendererChecks(cdp, label, runDir, mock) {
  const operations = []
  const measure = async (name, body) => {
    const started = performance.now()
    const value = await Promise.resolve().then(body)
    const latencyMs = performance.now() - started
    operations.push({ name, latencyMs })

    return value
  }
  const timed = (name, body) =>
    measure(name, () => withTimeout(Promise.resolve().then(body), FREEZE_MS, name, ReproductionError))

  await cdp.send('Runtime.enable')
  await cdp.send('Profiler.enable')
  await cdp.send('Profiler.start')
  await timed('harness.reset', () => cdp.eval('window.__SHORT_SESSION_HANG_REPRO__.reset()'))
  const realChat = await runRealChatChecks(cdp, timed, measure, mock, runDir)
  const fixtureManifest = await timed('fixture.manifest', () =>
    cdp.eval('window.__SHORT_SESSION_HANG_REPRO__.manifest()')
  )

  const checkpoints = []

  for (let count = 1; count <= 8; count += 1) {
    const loaded = await timed(`fixture.load.${count}`, () =>
      cdp.eval(`window.__SHORT_SESSION_HANG_REPRO__.load(${count})`)
    )
    await timed(`main.heartbeat.${count}`, () => cdp.eval('window.hermesDesktop.getVersion()'))

    if (count !== 5 && count !== 8) {
      continue
    }

    if (
      loaded.paintedUserIds.length !== loaded.expectedUserIds.length ||
      loaded.paintedAssistantMessages !== loaded.expectedAssistantMessages ||
      loaded.paintedTools !== loaded.expectedTools ||
      loaded.paintedCodeCards !== loaded.expectedCodeCards
    ) {
      throw new Error(`checkpoint ${count} did not paint the complete synthetic transcript: ${JSON.stringify(loaded)}`)
    }

    const surfaces = await verifyInteractiveSurfaces(cdp, timed, measure, `renderer-only-turn-${count}`)
    await withTimeout(
      screenshot(cdp, join(runDir, `renderer-only-turn-${count}.png`)),
      2_000,
      `checkpoint ${count} screenshot`
    )
    checkpoints.push({
      messageRecords: loaded.messageRecords,
      surfaces,
      syntheticTurns: loaded.syntheticTurns
    })
  }

  const heartbeat = await timed('renderer.heartbeat.summary', () =>
    cdp.eval('window.__SHORT_SESSION_HANG_REPRO__.summary()')
  )
  const profile = await cdp.send('Profiler.stop')
  writeFileSync(join(runDir, 'renderer.cpuprofile'), JSON.stringify(profile.profile))

  const maxOperationMs = operations.reduce((max, operation) => Math.max(max, operation.latencyMs), 0)
  const maxGapMs = Number(heartbeat.maxGapMs || 0)
  const reproduced = maxOperationMs > FREEZE_MS || maxGapMs > FREEZE_MS

  return {
    checkpoints,
    fixtureManifest,
    hardFailure: reproduced,
    maxGapMs,
    maxOperationMs,
    operations,
    outcome: reproduced ? 'reproduced' : 'not-reproduced',
    realChat,
    syntheticScenario: 'renderer-only'
  }
}

async function withTemporarySandbox(label, body) {
  const sandbox = mkdtempSync(join(tmpdir(), `hermes-short-session-${label}-`))

  try {
    return await body(sandbox)
  } finally {
    rmSync(sandbox, { force: true, recursive: true })
  }
}

function resultForError(error) {
  const reproduced = error instanceof ReproductionError

  return {
    error: error instanceof Error ? (error.stack ?? error.message) : String(error),
    hardFailure: true,
    lifecycleSignals: [],
    maxGapMs: null,
    maxOperationMs: null,
    operations: [],
    outcome: reproduced ? 'reproduced' : 'harness-error'
  }
}

async function executeRun(target, index, warmup, mock, output) {
  return withTemporarySandbox(target.label, sandbox =>
    executeRunInSandbox(target, index, warmup, mock, output, sandbox)
  )
}

async function executeRunInSandbox(target, index, warmup, mock, output, sandbox) {
  const runDir = join(output, target.label, warmup ? 'warmup' : `run-${index + 1}`)
  const hermesHome = join(sandbox, 'hermes-home')
  const userData = join(sandbox, 'electron-user-data')
  const desktopLog = join(hermesHome, 'logs', 'desktop.log')
  const stdoutPath = join(runDir, 'electron.stdout.log')
  const stderrPath = join(runDir, 'electron.stderr.log')
  const eventsPath = join(runDir, 'events.jsonl')
  const port = 19_000 + (process.pid % 1_000) + index * 4 + (target.label === 'candidate' ? 2 : 0) + (warmup ? 1 : 0)

  mkdirSync(runDir, { recursive: true })
  mkdirSync(userData, { recursive: true })
  writeSandboxConfig(hermesHome, mock.url)

  const electron = require('electron')
  const child = spawn(
    electron,
    [
      target.targetDesktop,
      `--user-data-dir=${userData}`,
      `--remote-debugging-port=${port}`,
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows'
    ],
    {
      cwd: target.targetDesktop,
      env: sanitizedEnv({
        HERMES_DESKTOP_APP_NAME: `HermesShortSession-${target.label}-${process.pid}-${index}-${warmup ? 'w' : 'm'}`,
        HERMES_DESKTOP_HERMES_ROOT: target.targetRoot,
        HERMES_DESKTOP_IGNORE_EXISTING: '1',
        HERMES_DESKTOP_USER_DATA_DIR: userData,
        HERMES_HOME: hermesHome,
        SHORT_SESSION_API_KEY: 'local-diagnostic-only'
      }),
      stdio: ['ignore', 'pipe', 'pipe']
    }
  )

  let exited = null
  let spawnFailed = false
  let stopping = false
  let resolveSpawn
  let rejectSpawn
  const spawnReady = new Promise((resolve, reject) => {
    resolveSpawn = resolve
    rejectSpawn = reject
  })
  child.once('spawn', () => resolveSpawn())
  child.once('error', error => {
    spawnFailed = true
    exited = exited ?? { code: null, signal: null }
    appendFileSync(
      eventsPath,
      `${JSON.stringify({ at: new Date().toISOString(), error: error.message, type: 'spawn-error' })}\n`
    )
    rejectSpawn(error)
  })
  child.stdout.on('data', chunk => appendFileSync(stdoutPath, chunk))
  child.stderr.on('data', chunk => appendFileSync(stderrPath, chunk))
  child.once('exit', (code, signal) => {
    exited = { code, signal }
    appendFileSync(
      eventsPath,
      `${JSON.stringify({ at: new Date().toISOString(), code, signal, type: stopping ? 'teardown-exit' : 'unexpected-exit' })}\n`
    )
  })

  let cdp
  let result

  try {
    await withTimeout(spawnReady, FREEZE_MS, 'Electron spawn')
    const targetInfo = await discoverTarget({ port, timeoutMs: 90_000 })
    cdp = await CDP.open(targetInfo.webSocketDebuggerUrl)
    await waitFor(cdp, '!!window.__SHORT_SESSION_HANG_REPRO__', 90_000, 'short-session renderer harness')
    await waitFor(
      cdp,
      `document.querySelector('[data-slot="composer-rich-input"]')?.contentEditable === 'true'`,
      90_000,
      'interactive composer'
    )

    result = await withWatchdog(runRendererChecks(cdp, `${target.label}-${index + 1}`, runDir, mock), () => {
      processSnapshot(join(runDir, 'hard-timeout-processes.txt'), child.pid)
      sampleTree(runDir, child.pid, 'hard-timeout')
    })
  } catch (error) {
    result = resultForError(error)
    processSnapshot(join(runDir, 'failure-processes.txt'), child.pid)
    sampleTree(runDir, child.pid, 'failure')
    if (cdp) {
      await withTimeout(screenshot(cdp, join(runDir, 'failure.png')), 2_000, 'failure screenshot').catch(() => {})
    }
  } finally {
    cdp?.close()
    const logText = existsSync(desktopLog) ? readFileSync(desktopLog, 'utf8') : ''
    const lifecycleSignals = logText
      .split(/\r?\n/)
      .filter(line => /webContents became unresponsive|render-process-gone/i.test(line))
    const lifecycleReproduced = lifecycleSignals.length > 0 || Boolean(exited && !spawnFailed)

    result.lifecycleSignals = lifecycleSignals

    if (lifecycleReproduced) {
      result.hardFailure = true
      result.outcome = 'reproduced'
    }

    stopping = true
    const teardown = await stopProcessTree(child.pid)
    writeFileSync(join(runDir, 'teardown.json'), `${JSON.stringify(teardown, null, 2)}\n`)

    if (existsSync(desktopLog)) {
      copyFileSync(desktopLog, join(runDir, 'desktop.log'))
    }

    if (teardown.remainingAfterKill.length > 0) {
      result = {
        ...(result ?? {}),
        error: `${result?.error ? `${result.error}\n` : ''}process teardown leaked PIDs ${teardown.remainingAfterKill.join(', ')}`,
        hardFailure: true,
        outcome: result?.outcome === 'reproduced' ? 'reproduced' : 'harness-error'
      }
    }
  }

  writeFileSync(join(runDir, 'result.json'), `${JSON.stringify(result, null, 2)}\n`)
  appendFileSync(eventsPath, `${JSON.stringify({ at: new Date().toISOString(), result, type: 'run-complete' })}\n`)

  return result
}

function classify(results, warmup) {
  const invalid = [warmup, ...results].some(result => result.outcome === 'harness-error')
  const reproduced = results.filter(result => result.outcome === 'reproduced').length
  const reproducedThreshold = Math.ceil(results.length * 0.8)

  return {
    classification: invalid
      ? 'invalid'
      : reproduced >= reproducedThreshold
        ? 'reproduced'
        : reproduced === 0
          ? 'not-reproduced'
          : 'intermittent',
    invalid,
    reproduced,
    reproducedThreshold,
    total: results.length
  }
}

function pairedSoftSignal(baseline, candidate) {
  if (baseline.length === 0 || candidate.length === 0) {
    return { material: false, materialPairs: 0, materialThreshold: 0, reason: 'insufficient-runs', thresholdPct: 30 }
  }

  if ([...baseline, ...candidate].some(result => result.outcome !== 'not-reproduced')) {
    return { material: false, materialPairs: 0, materialThreshold: 0, reason: 'hard-or-invalid-run', thresholdPct: 30 }
  }

  let materialPairs = 0

  for (let i = 0; i < Math.min(baseline.length, candidate.length); i += 1) {
    const base = Math.max(1, baseline[i].maxGapMs || 0, baseline[i].maxOperationMs || 0)
    const next = Math.max(candidate[i].maxGapMs || 0, candidate[i].maxOperationMs || 0)

    if (next >= base * 1.3) {
      materialPairs += 1
    }
  }

  const materialThreshold = Math.ceil(Math.min(baseline.length, candidate.length) * 0.8)

  return { material: materialPairs >= materialThreshold, materialPairs, materialThreshold, thresholdPct: 30 }
}

function validateArtifactBundle(output, repetitions) {
  const required = ['environment.json', 'summary.json', 'baseline-build.log', 'candidate-build.log']

  for (const label of ['baseline', 'candidate']) {
    for (const runName of ['warmup', ...Array.from({ length: repetitions }, (_, index) => `run-${index + 1}`)]) {
      for (const artifact of ['events.jsonl', 'result.json', 'teardown.json']) {
        required.push(join(label, runName, artifact))
      }
    }
  }

  for (const relativePath of required) {
    const path = join(output, relativePath)

    if (!existsSync(path) || readFileSync(path).byteLength === 0) {
      throw new Error(`missing or empty required diagnostic artifact: ${relativePath}`)
    }
  }

  const summary = JSON.parse(readFileSync(join(output, 'summary.json'), 'utf8'))

  validateSummary(summary, repetitions)
}

function validateSummary(summary, repetitions) {
  if (!Number.isInteger(repetitions) || repetitions < 1 || repetitions > 20) {
    throw new Error('invalid summary repetitions')
  }

  let invalid = false

  for (const label of ['baseline', 'candidate']) {
    if (!summary[label]?.warmup || summary[label].runs?.length !== repetitions) {
      throw new Error(`invalid ${label} summary shape`)
    }

    for (const result of [summary[label].warmup, ...summary[label].runs]) {
      if (!['harness-error', 'not-reproduced', 'reproduced'].includes(result.outcome)) {
        throw new Error(`invalid ${label} run outcome`)
      }

      const expectedHardFailure = result.outcome !== 'not-reproduced'

      if (result.hardFailure !== expectedHardFailure) {
        throw new Error(`inconsistent ${label} run outcome and hardFailure`)
      }
    }

    const derived = classify(summary[label].runs, summary[label].warmup)

    for (const field of ['classification', 'invalid', 'reproduced', 'reproducedThreshold', 'total']) {
      if (summary[label][field] !== derived[field]) {
        throw new Error(`inconsistent ${label} summary ${field}`)
      }
    }

    invalid ||= derived.invalid
  }

  if (summary.invalid !== invalid) {
    throw new Error('inconsistent summary invalid state')
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2))

  if (!existsSync(HARNESS_SOURCE)) {
    throw new Error(`renderer harness missing: ${HARNESS_SOURCE}`)
  }

  validateRef(options.baseline)
  validateRef(options.candidate)
  const baselineSha = resolveRef(options.baseline)
  const candidateSha = resolveRef(options.candidate)
  const mergeBase = run('git', ['merge-base', '--', baselineSha, candidateSha])
  const ancestor = spawnSync('git', ['merge-base', '--is-ancestor', '--', baselineSha, candidateSha], {
    cwd: REPO_ROOT,
    encoding: 'utf8'
  })

  if (mergeBase !== baselineSha || ancestor.status !== 0) {
    throw new Error(
      `baseline must be the exact merge-base and an ancestor of candidate; baseline=${baselineSha} merge-base=${mergeBase} candidate=${candidateSha}`
    )
  }

  const baselineMetadata = readTargetMetadata(baselineSha)
  const candidateMetadata = readTargetMetadata(candidateSha)
  const harnessHead = run('git', ['rev-parse', '--verify', '--end-of-options', 'HEAD^{commit}'])
  const harnessMetadata = readTargetMetadata(harnessHead)

  if (
    JSON.stringify(baselineMetadata) !== JSON.stringify(candidateMetadata) ||
    JSON.stringify(baselineMetadata) !== JSON.stringify(harnessMetadata)
  ) {
    throw new Error(
      `shared dependency mismatch; harness checkout, baseline, and candidate Electron/lockfile metadata must match:\nharness ${JSON.stringify(harnessMetadata)}\nbaseline ${JSON.stringify(baselineMetadata)}\ncandidate ${JSON.stringify(candidateMetadata)}`
    )
  }

  const plan = {
    baseline: { ref: options.baseline, sha: baselineSha },
    candidate: { ref: options.candidate, sha: candidateSha },
    harness: { head: harnessHead, metadata: harnessMetadata },
    harnessSha256: sha256(readFileSync(HARNESS_SOURCE)),
    mergeBase,
    metadata: baselineMetadata,
    repetitions: options.repetitions,
    runner: { arch: process.arch, platform: process.platform, versions: process.versions }
  }

  if (options.dryRun) {
    console.log(JSON.stringify(plan, null, 2))

    return
  }

  if (process.platform !== 'darwin' || process.arch !== 'arm64') {
    throw new Error(
      `the short-session hang diagnostic requires macOS arm64; got ${process.platform}-${process.arch} (use --dry-run for preflight)`
    )
  }

  mkdirSync(options.output, { recursive: true })
  writeFileSync(join(options.output, 'environment.json'), `${JSON.stringify(plan, null, 2)}\n`)

  const ephemeralRoot = mkdtempSync(join(tmpdir(), 'hermes-short-session-ab-'))
  const prepared = []
  const mock = await startMockInference()

  try {
    prepared.push(prepareTarget('baseline', baselineSha, ephemeralRoot, options.output))
    prepared.push(prepareTarget('candidate', candidateSha, ephemeralRoot, options.output))
    const byLabel = Object.fromEntries(prepared.map(target => [target.label, target]))

    const warmups = {}

    for (const label of ['baseline', 'candidate']) {
      warmups[label] = await executeRun(byLabel[label], 0, true, mock, options.output)
    }

    const measured = { baseline: [], candidate: [] }

    for (let index = 0; index < options.repetitions; index += 1) {
      const order = index % 2 === 0 ? ['baseline', 'candidate'] : ['candidate', 'baseline']

      for (const label of order) {
        measured[label].push(await executeRun(byLabel[label], index, false, mock, options.output))
      }
    }

    const baselineClassification = classify(measured.baseline, warmups.baseline)
    const candidateClassification = classify(measured.candidate, warmups.candidate)
    const invalid = baselineClassification.invalid || candidateClassification.invalid
    const summary = {
      ...plan,
      invalid,
      baseline: { ...plan.baseline, ...baselineClassification, runs: measured.baseline, warmup: warmups.baseline },
      candidate: {
        ...plan.candidate,
        ...candidateClassification,
        runs: measured.candidate,
        warmup: warmups.candidate
      },
      softSignal: invalid
        ? { material: false, materialPairs: 0, materialThreshold: 0, reason: 'invalid-run', thresholdPct: 30 }
        : pairedSoftSignal(measured.baseline, measured.candidate)
    }
    writeFileSync(join(options.output, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`)
    validateArtifactBundle(options.output, options.repetitions)
    console.log(JSON.stringify(summary, null, 2))

    if (
      warmups.baseline.hardFailure ||
      warmups.candidate.hardFailure ||
      measured.baseline.some(result => result.hardFailure) ||
      measured.candidate.some(result => result.hardFailure)
    ) {
      process.exitCode = 1
    }
  } finally {
    await mock.close()

    if (!options.keepWorktrees) {
      for (const target of prepared.reverse()) {
        try {
          run('git', ['worktree', 'remove', '--force', target.targetRoot])
        } catch (error) {
          console.error(`warning: failed to remove ${target.targetRoot}: ${error.message}`)
        }
      }

      rmSync(ephemeralRoot, { force: true, recursive: true })
    } else {
      console.log(`kept ephemeral worktrees at ${ephemeralRoot}`)
    }
  }
}

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : String(error))
    process.exitCode = 2
  })
}

export {
  ReproductionError,
  classify,
  pairedSoftSignal,
  resultForError,
  validateArtifactBundle,
  validateSummary,
  waitFor,
  waitForPredicate,
  waitForResponsive,
  withTemporarySandbox,
  withTimeout
}
