export { CoderPromptBuilder, CoderPromptOptions, WorkerPromptBuilder, WorkerPromptOptions, stripLibrarianReferences } from './workerPrompt';
export { LibrarianPromptBuilder } from './librarianPrompt';
export { OrchestratorPromptBuilder, OrchestratorPromptOptions } from './orchestratorPrompt';

// ─── Agent Mode Prompt Variant (Phase 5: agent embedded in assistant mode) ──
// Agent mode is a CAPABILITY LAYER appended to the active personality prompt.
// It is NOT a separate persona and must NEVER trigger a UI switch: the chat
// surface stays 100% identical to assistant mode. The only difference is that
// the LLM may read, write and execute code in rounds that are ALL integrated
// inside the same answer block.
export interface AgentModeLayerOptions {
    enabledTools?: string[];
    maxRounds?: number;
}

export const AGENT_MODE_PROMPT_HEADER = '## 🤖 Agent Mode Capability Layer';

export function buildAgentModePromptLayer(basePrompt: string, options: AgentModeLayerOptions = {}): string {
    if (!basePrompt || !basePrompt.trim()) {
        return basePrompt;
    }
    const tools = (options.enabledTools && options.enabledTools.length)
        ? options.enabledTools.map((t: string) => `- ${t}`).join('\n')
        : '- read_files\n- write_files\n- execute_code';
    const maxRounds = options.maxRounds ?? 12;
    return `${basePrompt.trimEnd()}

${AGENT_MODE_PROMPT_HEADER}
You are operating in AGENT MODE within the SAME assistant chat: identical personality, identical formatting style, identical user experience. The only difference: you may autonomously read, write, and execute code.
- Available tools:
${tools}
- Work in ROUNDS (plan → tool use → execution → verification). Every round MUST be emitted inside the SAME answer block using fenced markers: <agent_round type="plan|tool|exec|verify|question" title="..." status="running|done|failed">…</agent_round>. Never split rounds across separate answer messages.
- After at most ${maxRounds} consecutive autonomous rounds, surface control back to the user with a clear summary.
- All base-prompt safety rules remain fully in force.
`;
}