import { bindManual } from "./support_manual.js";
import { mountLlmConsole } from "./support_llm.js";

bindManual("manualBtn", "manualModal");
mountLlmConsole("llmConfigPanel", {promptKey: "planner.dynamic_tool_summary"});
