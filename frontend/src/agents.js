// The 7-agent architecture, mirroring agents/orchestrator/graph.py.
// `llm: true` agents are not run in the dry pipeline (Step 1) yet.

export const AGENTS = [
  { key: 'ingestor',       name: 'Data Ingestor',       tier: 1, icon: '📥', desc: 'Validates & structures the raw payload' },
  { key: 'sensor_parser',  name: 'Sensor Parser',       tier: 1, icon: '📈', desc: 'Derives metrics, flags failure risks' },
  { key: 'classifier',     name: 'Defect Classifier',   tier: 2, icon: '🧠', desc: 'XGBoost defect prediction' },
  { key: 'root_cause',     name: 'Root Cause Analyzer', tier: 2, icon: '🔍', desc: 'RAG + GPT-4o causal analysis', llm: true },
  { key: 'trend',          name: 'Trend Detector',      tier: 2, icon: '📊', desc: 'Batch / machine pattern detection', llm: true },
  { key: 'supplier_alert', name: 'Supplier Alert',      tier: 3, icon: '🚨', desc: 'Notifies supplier on material defects', llm: true },
  { key: 'compliance',     name: 'Compliance Report',   tier: 3, icon: '📋', desc: 'Generates ISO 9001 report', llm: true },
]

export const TIERS = {
  1: 'Tier 1 · Intake',
  2: 'Tier 2 · Analysis',
  3: 'Tier 3 · Action',
}

// Derive an agent's final status from the returned PipelineState.
// Returns one of: 'done' | 'failed' | 'skipped'
export function finalStatusFor(key, state) {
  if (!state) return 'skipped'
  const failed = state.status === 'failed'
  switch (key) {
    case 'ingestor':       return state.ingested ? 'done' : 'failed'
    case 'sensor_parser':  return state.sensor_summary ? 'done' : 'skipped'
    case 'classifier':     return state.classification ? 'done' : (failed ? 'skipped' : 'skipped')
    case 'root_cause':     return state.root_cause ? 'done' : 'skipped'
    case 'trend':          return state.trend ? 'done' : 'skipped'
    case 'supplier_alert': return state.supplier_alert ? 'done' : 'skipped'
    case 'compliance':     return state.compliance_report ? 'done' : 'skipped'
    default:               return 'skipped'
  }
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
