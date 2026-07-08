#!/usr/bin/env node
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const PROVIDER_ID = process.env.SAFACTORY_OPENCLAW_PROVIDER_ID || "safactory";
const DEFAULT_HOST_GATEWAY = process.env.SAFACTORY_DOCKER_HOST_GATEWAY || "host.docker.internal";
const DIAGNOSTIC_PREFIX = "SAFACTORY_RUNNER_DIAGNOSTIC ";

async function main() {
  const startedAt = Date.now();
  const request = JSON.parse(await readStdin());
  const sessionId = requiredString(request.session_id, "session_id");
  const routeModel = requiredString(request.model, "model");
  const modelRef = `${PROVIDER_ID}/${routeModel}`;
  const gatewaySessionBaseUrl = gatewayBaseUrlForContainer(request.gateway_base_url, sessionId);
  const stateDir = join("/tmp", "safactory-openclaw", sessionId);
  const configPath = join(stateDir, "openclaw.json");
  const timeoutSeconds = Math.max(1, Number(request.agent_start_timeout_s || request.timeout_s || 600));

  await mkdir(stateDir, { recursive: true });
  await writeOpenClawConfig({
    configPath,
    providerBaseUrl: gatewaySessionBaseUrl,
    routeModel,
    modelRef,
    timeoutSeconds,
  });

  const message = buildAgentMessage(request);
  const args = [
    "agent",
    "--local",
    "--json",
    "--session-id",
    sessionId,
    "--message",
    message,
    "--model",
    modelRef,
    "--timeout",
    String(timeoutSeconds),
  ];

  const openClawEnv = {
    ...process.env,
    NO_COLOR: "1",
    OPENCLAW_CONFIG_PATH: configPath,
    OPENCLAW_STATE_DIR: stateDir,
    OPENCLAW_PROFILE: "",
  };
  logOpenClawCreateParams({
    command: ["openclaw", ...args],
    command_string: shellQuoteCommand(["openclaw", ...args]),
    args,
    config_path: configPath,
    state_dir: stateDir,
    provider_id: PROVIDER_ID,
    route_model: routeModel,
    model_ref: modelRef,
    gateway_session_base_url: gatewaySessionBaseUrl,
    timeout_seconds: timeoutSeconds,
    env: {
      NO_COLOR: openClawEnv.NO_COLOR,
      OPENCLAW_CONFIG_PATH: openClawEnv.OPENCLAW_CONFIG_PATH,
      OPENCLAW_STATE_DIR: openClawEnv.OPENCLAW_STATE_DIR,
      OPENCLAW_PROFILE: openClawEnv.OPENCLAW_PROFILE,
    },
    request,
  });

  const run = await runOpenClaw(args, openClawEnv);

  const elapsedMs = Date.now() - startedAt;
  const parsed = parseJsonFromText(run.stdout);
  const metrics = {
    duration_ms: elapsedMs,
    openclaw_exit_code: run.code,
    openclaw_model_ref: modelRef,
    gateway_session_base_url: gatewaySessionBaseUrl,
  };
  if (parsed !== undefined) {
    metrics.openclaw_result = summarizeOpenClawResult(parsed);
  }

  const openClawFailure = detectOpenClawFailure(parsed, run);
  if (run.code !== 0 || openClawFailure) {
    metrics.stdout_tail = tail(run.stdout, 2000);
    metrics.stderr_tail = tail(run.stderr, 3000);
  } else if (String(run.stderr || "").trim()) {
    metrics.stderr_tail = tail(run.stderr, 1000);
  }

  if (run.code !== 0 || openClawFailure) {
    writeResult({
      session_id: sessionId,
      status: "failed",
      total_reward: 0,
      step_count: openClawFailure ? 1 : 0,
      terminated: true,
      truncated: false,
      error_text: buildErrorText(run, parsed),
      metrics,
    });
    return;
  }

  writeResult({
    session_id: sessionId,
    status: "succeeded",
    total_reward: inferReward(parsed),
    step_count: inferStepCount(parsed),
    terminated: true,
    truncated: false,
    error_text: null,
    metrics,
  });
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let body = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      body += chunk;
    });
    process.stdin.on("end", () => resolve(body));
    process.stdin.on("error", reject);
  });
}

function requiredString(value, name) {
  const text = String(value || "").trim();
  if (!text) {
    throw new Error(`SimulationStartRequest missing ${name}`);
  }
  return text;
}

function gatewayBaseUrlForContainer(rawBaseUrl, sessionId) {
  const url = new URL(requiredString(rawBaseUrl, "gateway_base_url"));
  const host = url.hostname.toLowerCase();
  if (host === "127.0.0.1" || host === "localhost" || host === "0.0.0.0" || host === "::1") {
    url.hostname = DEFAULT_HOST_GATEWAY;
  }
  url.pathname = `${url.pathname.replace(/\/+$/, "")}/${encodeURIComponent(sessionId)}`;
  url.search = "";
  url.hash = "";
  return url.toString().replace(/\/+$/, "");
}

async function writeOpenClawConfig({ configPath, providerBaseUrl, routeModel, modelRef, timeoutSeconds }) {
  const cfg = {
    env: {
      shellEnv: {
        enabled: false,
        timeoutMs: 1000,
      },
    },
    agents: {
      defaults: {
        model: {
          primary: modelRef,
          timeoutMs: timeoutSeconds * 1000,
        },
      },
    },
    models: {
      pricing: {
        enabled: false,
      },
      providers: {
        [PROVIDER_ID]: {
          baseUrl: providerBaseUrl,
          apiKey: process.env.SAFACTORY_GATEWAY_API_KEY || "safactory",
          api: "openai-completions",
          request: {
            allowPrivateNetwork: true,
          },
          timeoutSeconds,
          models: [
            {
              id: routeModel,
              name: `Safactory Gateway ${routeModel}`,
              api: "openai-completions",
              reasoning: false,
              input: ["text", "image"],
              cost: {
                input: 0,
                output: 0,
                cacheRead: 0,
                cacheWrite: 0,
              },
              contextWindow: 128000,
              maxTokens: 8192,
            },
          ],
        },
      },
    },
  };
  await writeFile(configPath, `${JSON.stringify(cfg, null, 2)}\n`, "utf8");
}

function buildAgentMessage(request) {
  const envParams = objectOrEmpty(request.env_params);
  const metadata = objectOrEmpty(request.metadata);
  const taskText = pickTaskText(envParams) || pickTaskText(metadata) || "";
  const taskFamily = envParams.task_family || metadata.task_family || request.agent_name || "openclaw";
  const parts = [
    `You are running a Safactory ${taskFamily} episode.`,
    `Session id: ${request.session_id || ""}`,
    `Maximum steps: ${request.max_steps || ""}`,
  ];

  if (taskText) {
    parts.push(`Task:\n${taskText}`);
  }

  parts.push(
    "Use the available OpenClaw tools to complete the task. Keep the final answer concise and report the outcome.",
  );

  if (Object.keys(envParams).length > 0) {
    parts.push(`Safactory environment parameters:\n${JSON.stringify(envParams, null, 2)}`);
  }
  return parts.filter(Boolean).join("\n\n");
}

function objectOrEmpty(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function pickTaskText(root) {
  const keys = [
    "task",
    "instruction",
    "instructions",
    "prompt",
    "query",
    "question",
    "goal",
    "objective",
    "description",
  ];
  for (const key of keys) {
    if (typeof root[key] === "string" && root[key].trim()) {
      return root[key].trim();
    }
  }

  const dataset = objectOrEmpty(root.dataset);
  for (const key of keys) {
    if (typeof dataset[key] === "string" && dataset[key].trim()) {
      return dataset[key].trim();
    }
  }

  if (Array.isArray(dataset.messages)) {
    return dataset.messages
      .map((message) => {
        if (typeof message === "string") {
          return message;
        }
        if (message && typeof message === "object") {
          const role = message.role ? `${message.role}: ` : "";
          return `${role}${message.content || ""}`.trim();
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function logOpenClawCreateParams(value) {
  process.stderr.write(`${DIAGNOSTIC_PREFIX}${JSON.stringify(value)}\n`);
}

function shellQuoteCommand(parts) {
  return parts.map(shellQuote).join(" ");
}

function shellQuote(value) {
  const text = String(value);
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) {
    return text;
  }
  return `'${text.replace(/'/g, "'\\''")}'`;
}

function runOpenClaw(args, env) {
  return new Promise((resolve) => {
    const child = spawn("openclaw", args, {
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      resolve({ code: 127, stdout, stderr: `${stderr}\n${error.stack || error.message}` });
    });
    child.on("close", (code) => {
      resolve({ code: Number(code || 0), stdout, stderr });
    });
  });
}

function parseJsonFromText(text) {
  const body = String(text || "").trim();
  if (!body) {
    return undefined;
  }
  try {
    return JSON.parse(body);
  } catch {
    for (const line of body.split(/\r?\n/).reverse()) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      try {
        return JSON.parse(trimmed);
      } catch {
        continue;
      }
    }
  }
  return undefined;
}

function inferReward(parsed) {
  if (parsed && typeof parsed === "object") {
    if (typeof parsed.total_reward === "number") {
      return parsed.total_reward;
    }
    if (typeof parsed.reward === "number") {
      return parsed.reward;
    }
  }
  return 0;
}

function inferStepCount(parsed) {
  if (parsed && typeof parsed === "object") {
    if (Number.isFinite(Number(parsed.step_count))) {
      return Math.max(1, Number(parsed.step_count));
    }
    if (Array.isArray(parsed.messages)) {
      return Math.max(1, parsed.messages.length);
    }
  }
  return 1;
}

function detectOpenClawFailure(parsed, run) {
  if (parsed && typeof parsed === "object") {
    if (parsed.error || parsed.isError === true) {
      return true;
    }
    if (parsed.meta && typeof parsed.meta === "object" && parsed.meta.aborted === true) {
      return true;
    }
    const payloadText = payloadTextFromResult(parsed).toLowerCase();
    if (payloadText.includes("request timed out") || payloadText.includes("before a response was generated")) {
      return true;
    }
  }
  const stderr = String(run.stderr || "");
  return stderr.includes("isError=true") || stderr.includes("FailoverError:");
}

function summarizeOpenClawResult(parsed) {
  if (!parsed || typeof parsed !== "object") {
    return parsed;
  }
  const meta = parsed.meta && typeof parsed.meta === "object" ? parsed.meta : {};
  return {
    payloads: Array.isArray(parsed.payloads) ? parsed.payloads.slice(0, 3) : undefined,
    error: parsed.error,
    isError: parsed.isError,
    meta: {
      durationMs: meta.durationMs,
      aborted: meta.aborted,
      livenessState: meta.livenessState,
      agentMeta: meta.agentMeta,
    },
  };
}

function payloadTextFromResult(parsed) {
  if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.payloads)) {
    return "";
  }
  return parsed.payloads
    .map((payload) => (payload && typeof payload === "object" ? payload.text || "" : ""))
    .join("\n");
}

function buildErrorText(run, parsed) {
  const payloadText = tail(payloadTextFromResult(parsed), 2000);
  const stderr = tail(run.stderr, 3000);
  const stdout = payloadText ? "" : tail(run.stdout, 500);
  return `OpenClaw agent failed: returncode=${run.code} payload=${payloadText} stdout=${stdout} stderr=${stderr}`.trim();
}

function tail(value, limit) {
  const text = String(value || "").trim();
  return text.slice(Math.max(0, text.length - limit));
}

function writeResult(result) {
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

main().catch((error) => {
  writeResult({
    session_id: process.env.SAFACTORY_SESSION_ID || "",
    status: "failed",
    total_reward: 0,
    step_count: 0,
    terminated: true,
    truncated: false,
    error_text: error.stack || error.message || String(error),
    metrics: {},
  });
});
