// Per-agent output cards — what each agent actually contributed to the run.

const SEVERITY_CLASS = { low: 'sev-low', medium: 'sev-medium', high: 'sev-high', critical: 'sev-critical' }
const yn = (b) => (b ? 'Yes' : 'No')

export default function AgentDetails({ state }) {
  const insp = state.ingested?.inspection
  const clf = state.classification
  const summary = state.sensor_summary
  const rc = state.root_cause
  const trend = state.trend
  const supplier = state.supplier_alert
  const compliance = state.compliance_report

  return (
    <div className="details">
      {/* 01 Ingestor */}
      <OutputCard no="01" title="Data Ingestor" ran={!!state.ingested}>
        {insp && (
          <dl className="kv">
            <Row k="Wafer" v={insp.wafer_id || insp.record_id} />
            <Row k="Lot" v={insp.batch_id} />
            <Row k="Tool" v={insp.machine_id} />
            <Row k="Process data" v={insp.process_sensors ? `${insp.process_sensors.length} sensors` : 'none'} />
          </dl>
        )}
        {state.ingested?.ingest_warnings?.length > 0 && (
          <details className="warnings">
            <summary>{state.ingested.ingest_warnings.length} note(s)</summary>
            <ul>{state.ingested.ingest_warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
          </details>
        )}
      </OutputCard>

      {/* 02 Sensor Parser */}
      <OutputCard no="02" title="Sensor Parser" ran={!!summary}>
        {summary && (
          <>
            <dl className="kv">
              <Row k="Yield" v={`${summary.wafer.yield_pct}% (${summary.wafer.failing_die} of ${summary.wafer.die} die fail)`} />
              <Row k="SPC" v={summary.spc
                ? `${summary.spc.out_of_control} of ${summary.spc.sensors_checked} sensors outside 3σ (normal ≤${Math.round(summary.spc.typical_out_of_control_p95)})`
                : 'no process data'} />
              <Row k="Run unusual?" v={yn(summary.any_anomaly)} />
            </dl>
            {summary.spc?.worst_sensors?.length > 0 && (
              <p className="batch-hint">Furthest out: {summary.spc.worst_sensors.map((w) => `${w.sensor} ${w.z_score > 0 ? '+' : ''}${w.z_score}σ`).join(', ')}</p>
            )}
          </>
        )}
      </OutputCard>

      {/* 03 Classifier */}
      <OutputCard no="03" title="Defect Classifier" ran={!!clf} tag="ML">
        {clf && (
          <>
            <div className="clf-headline">
              <span className={`badge ${SEVERITY_CLASS[clf.severity] || ''}`}>{clf.severity}</span>
              <span className="defect">{clf.pattern}</span>
              <span className="conf">{Math.round((clf.confidence_score || 0) * 100)}%</span>
            </div>
            <dl className="kv">
              <Row k="Where" v={clf.location_description} />
              <Row k="Runner-up" v={clf.runner_up ? `${clf.runner_up.pattern} (${Math.round(clf.runner_up.probability * 100)}%)` : '—'} />
              <Row k="Action" v={clf.recommended_action} />
              <Row k="Human review" v={clf.requires_human_review ? 'Required' : 'No'} />
            </dl>
          </>
        )}
      </OutputCard>

      {/* 04 Root Cause */}
      <OutputCard no="04" title="Root Cause Analyzer" ran={!!rc} tag="LLM">
        {rc && (
          <>
            {rc.primary_cause && <p className="prose"><strong>{rc.primary_cause}</strong></p>}
            {rc.explanation && <p className="prose">{rc.explanation}</p>}
            {rc.contributing_factors?.length > 0 && (
              <ul className="bullets">{rc.contributing_factors.map((f, i) => <li key={i}>{f}</li>)}</ul>
            )}
            <dl className="kv">
              <Row k="Similar wafers agree" v={rc.neighbour_agreement || '—'} />
              <Row k="Material / machine / process" v={`${yn(rc.material_issue)} / ${yn(rc.machine_issue)} / ${yn(rc.process_issue)}`} />
            </dl>
            {rc.similar_past_defects?.length > 0 && (
              <details className="warnings">
                <summary>10 nearest historical wafers</summary>
                <ul>{rc.similar_past_defects.map((w, i) => <li key={i}>{w}</li>)}</ul>
              </details>
            )}
          </>
        )}
      </OutputCard>

      {/* 05 Trend Detector */}
      <OutputCard no="05" title="Trend Detector" ran={!!trend} tag="ML">
        {trend && (
          <>
            <dl className="kv">
              <Row k="Pattern" v={trend.pattern?.replaceAll('_', ' ')} />
              <Row k="Same pattern in lot" v={`${trend.lot_same_pattern} of ${trend.lot_wafers_seen} inspected`} />
              <Row k="Escalate" v={yn(trend.escalate)} />
              <Row k="Process risk*" v={trend.process_risk != null ? `${trend.process_risk.toFixed(2)}${trend.process_alert ? ' (above threshold)' : ''}` : '—'} />
            </dl>
            {trend.summary && <p className="prose trend-summary">{trend.summary}</p>}
          </>
        )}
      </OutputCard>

      {/* 06 Supplier Alert */}
      <OutputCard no="06" title="Supplier Alert" ran={!!supplier} tag="LLM">
        {supplier && (supplier.triggered ? (
          <div className="notice">
            <div className="notice-head">
              <span className={`badge ${SEVERITY_CLASS[supplier.severity] || ''}`}>{supplier.severity}</span>
              <strong>Alert sent to supplier</strong>
            </div>
            <dl className="kv">
              <Row k="To" v={supplier.supplier_id || '—'} />
              <Row k="Material batch" v={supplier.material_batch_id || '—'} />
              {supplier.sent_at && <Row k="Sent" v={new Date(supplier.sent_at + 'Z').toLocaleString()} />}
            </dl>
            <p className="notice-body">{supplier.alert_message}</p>
          </div>
        ) : (
          <>
            <dl className="kv"><Row k="Triggered" v="No" /></dl>
            <p className="prose">{supplier.alert_message}</p>
            {!insp?.supplier_id && (
              <p className="dry-note">No Supplier ID on this inspection, so no supplier could be alerted.</p>
            )}
          </>
        ))}
      </OutputCard>

      {/* 07 Compliance Report */}
      <OutputCard no="07" title="Compliance Report" ran={!!compliance} tag="LLM" wide>
        {compliance && (
          <>
            <dl className="kv">
              <Row k="Standard" v={compliance.iso_standard} />
              {compliance.report_id && <Row k="Report ID" v={compliance.report_id} />}
              <Row k="Sign-off" v={compliance.sign_off_required ? 'Required' : 'No'} />
            </dl>
            {compliance.full_report_text && (
              <>
                <pre className="report-text report-full">{compliance.full_report_text}</pre>
                <button type="button" className="link report-dl" onClick={() => downloadReport(compliance)}>
                  ↓ Download report (.txt)
                </button>
              </>
            )}
          </>
        )}
      </OutputCard>
    </div>
  )
}

function downloadReport(report) {
  const blob = new Blob([report.full_report_text], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${report.report_id || 'compliance-report'}.txt`
  a.click()
  URL.revokeObjectURL(url)
}

function OutputCard({ no, title, ran, tag, wide, children }) {
  return (
    <div className={`card output-card ${ran ? '' : 'inactive'} ${wide && ran ? 'wide' : ''}`}>
      <div className="output-head">
        <span className="output-no">{no}</span>
        <h3>{title}</h3>
        {tag && <span className={`llm-tag ${tag === 'ML' ? 'ml-tag' : ''}`}>{tag}</span>}
        <span className={`pip ${ran ? 'pip-on' : 'pip-off'}`}>{ran ? 'done' : 'idle'}</span>
      </div>
      {ran ? children : <div className="dry-note">Not run for this record.</div>}
    </div>
  )
}

function Row({ k, v }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  )
}
