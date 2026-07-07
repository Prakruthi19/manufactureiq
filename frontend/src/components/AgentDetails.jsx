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
            <Row k="Product" v={insp.product_id} />
            <Row k="Batch" v={insp.batch_id} />
            <Row k="Machine" v={insp.machine_id} />
            <Row k="Sensors" v={state.ingested.sensors?.length ?? 0} />
          </dl>
        )}
        {state.ingested?.ingest_warnings?.length > 0 && (
          <details className="warnings">
            <summary>{state.ingested.ingest_warnings.length} warning(s)</summary>
            <ul>{state.ingested.ingest_warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
          </details>
        )}
      </OutputCard>

      {/* 02 Sensor Parser */}
      <OutputCard no="02" title="Sensor Parser" ran={!!summary}>
        {summary && (
          <dl className="kv">
            <Row k="Anomaly" v={yn(summary.any_anomaly)} />
            <Row k="Risks" v={(summary.risk_flags || []).join(', ') || 'none'} />
            <Row k="Out of range" v={[
              summary.temp_out_of_range && 'temp',
              summary.rpm_out_of_range && 'rpm',
              summary.torque_out_of_range && 'torque',
            ].filter(Boolean).join(', ') || 'none'} />
          </dl>
        )}
      </OutputCard>

      {/* 03 Classifier */}
      <OutputCard no="03" title="Defect Classifier" ran={!!clf}>
        {clf && (
          <>
            <div className="clf-headline">
              <span className={`badge ${SEVERITY_CLASS[clf.severity] || ''}`}>{clf.severity}</span>
              <span className="defect">{clf.defect_type?.replaceAll('_', ' ')}</span>
              <span className="conf">{Math.round((clf.confidence_score || 0) * 100)}%</span>
            </div>
            <dl className="kv">
              <Row k="Location" v={clf.location_description} />
              <Row k="Action" v={clf.recommended_action} />
              <Row k="Review" v={clf.requires_human_review ? 'Required' : 'No'} />
            </dl>
          </>
        )}
      </OutputCard>

      {/* 04 Root Cause */}
      <OutputCard no="04" title="Root Cause Analyzer" ran={!!rc} llm>
        {rc && (
          <>
            {rc.primary_cause && <p className="prose">{rc.primary_cause}</p>}
            {rc.contributing_factors?.length > 0 && (
              <ul className="bullets">{rc.contributing_factors.map((f, i) => <li key={i}>{f}</li>)}</ul>
            )}
            <dl className="kv">
              <Row k="Material issue" v={yn(rc.material_issue)} />
              <Row k="Machine issue" v={yn(rc.machine_issue)} />
              <Row k="Process issue" v={yn(rc.process_issue)} />
            </dl>
          </>
        )}
      </OutputCard>

      {/* 05 Trend Detector */}
      <OutputCard no="05" title="Trend Detector" ran={!!trend} llm>
        {trend && (
          <dl className="kv">
            <Row k="Pattern" v={trend.pattern?.replaceAll('_', ' ')} />
            <Row k="Escalate" v={yn(trend.escalate)} />
            <Row k="Freq (7d)" v={trend.defect_frequency_last_7d} />
            {trend.summary && <Row k="Summary" v={trend.summary} />}
          </dl>
        )}
      </OutputCard>

      {/* 06 Supplier Alert */}
      <OutputCard no="06" title="Supplier Alert" ran={!!supplier} llm>
        {supplier && (
          <>
            <dl className="kv">
              <Row k="Triggered" v={yn(supplier.triggered)} />
              {supplier.triggered && <Row k="Severity" v={supplier.severity} />}
              {supplier.supplier_id && <Row k="Supplier" v={supplier.supplier_id} />}
            </dl>
            {supplier.alert_message && <p className="prose">{supplier.alert_message}</p>}
          </>
        )}
      </OutputCard>

      {/* 07 Compliance Report */}
      <OutputCard no="07" title="Compliance Report" ran={!!compliance} llm>
        {compliance && (
          <>
            <dl className="kv">
              <Row k="Standard" v={compliance.iso_standard} />
              {compliance.report_id && <Row k="Report ID" v={compliance.report_id} />}
              <Row k="Sign-off" v={compliance.sign_off_required ? 'Required' : 'No'} />
            </dl>
            {compliance.full_report_text && (
              <details className="warnings">
                <summary>View report</summary>
                <pre className="report-text">{compliance.full_report_text}</pre>
              </details>
            )}
          </>
        )}
      </OutputCard>
    </div>
  )
}

function OutputCard({ no, title, ran, llm, children }) {
  return (
    <div className={`card output-card ${ran ? '' : 'inactive'}`}>
      <div className="output-head">
        <span className="output-no">{no}</span>
        <h3>{title}</h3>
        {llm && <span className="llm-tag">LLM</span>}
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
