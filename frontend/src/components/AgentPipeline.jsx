import { Fragment } from 'react'
import ProgressBar from './ProgressBar'

// Horizontal multi-agent DAG: Tier 1 → Tier 2 → Tier 3, each node showing live status.

const STATE_LABEL = {
  queued: 'queued',
  running: 'running',
  done: 'done',
  skipped: 'not run',
  failed: 'failed',
}

export default function AgentPipeline({ agentStatus, tiers, agents, progress = 0 }) {
  // Global step numbers (01..07) across the whole pipeline.
  const stepOf = Object.fromEntries(agents.map((a, i) => [a.key, String(i + 1).padStart(2, '0')]))
  const byTier = { 1: [], 2: [], 3: [] }
  agents.forEach((a) => byTier[a.tier].push(a))
  const tierKeys = [1, 2, 3]

  return (
    <div className="card">
      <div className="card-head"><h2>Agent pipeline</h2></div>
      <ProgressBar value={progress} label="Pipeline progress" />
      <div className="pipeline-flow">
        {tierKeys.map((t, i) => (
          <Fragment key={t}>
            <div className="tier-col">
              <div className="tier-label">{tiers[t]}</div>
              <div className="tier-agents">
                {byTier[t].map((a) => (
                  <AgentNode key={a.key} agent={a} step={stepOf[a.key]} status={agentStatus[a.key] || 'queued'} />
                ))}
              </div>
            </div>
            {i < tierKeys.length - 1 && <div className="tier-arrow">→</div>}
          </Fragment>
        ))}
      </div>
      <p className="pipeline-caption">
        Full pipeline: 7 agents across 3 tiers. LLM agents (04, 06, 07) call GPT-4o, so a run takes ~15–40s.
      </p>
    </div>
  )
}

function AgentNode({ agent, step, status }) {
  return (
    <div className={`agent-node ${status}`}>
      <span className="agent-no">{step}</span>
      <div className="agent-body">
        <div className="agent-name">
          {agent.name}
          {agent.tag && <span className={`llm-tag ${agent.tag === 'ML' ? 'ml-tag' : ''}`}>{agent.tag}</span>}
        </div>
        <div className="agent-desc">{agent.desc}</div>
        {status === 'running' && <div className="node-bar"><span /></div>}
      </div>
      <span className="agent-state">
        {status === 'running' ? <span className="node-spinner" /> : STATE_LABEL[status]}
      </span>
    </div>
  )
}
