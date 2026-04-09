import { useEffect, useState } from "react";
import {
  listAlertRules,
  updateAlertRule,
  deleteAlertRule,
  getAlertHistory,
  listUseCaseTemplates,
  applyUseCaseTemplate,
  type AlertRule,
  type AlertHistoryEntry,
  type UseCaseTemplate,
} from "../api/client";

interface AlertsPanelProps {
  onBack: () => void;
}

const SEVERITY_COLORS: Record<string, string> = {
  low: "#94a3b8",
  medium: "#f59e0b",
  high: "#f97316",
  critical: "#dc2626",
};

export default function AlertsPanel({ onBack }: AlertsPanelProps) {
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [history, setHistory] = useState<AlertHistoryEntry[]>([]);
  const [templates, setTemplates] = useState<UseCaseTemplate[]>([]);
  const [tab, setTab] = useState<"rules" | "history" | "templates">("rules");

  const reload = async () => {
    try {
      const [r, h, t] = await Promise.all([
        listAlertRules(),
        getAlertHistory(),
        listUseCaseTemplates(),
      ]);
      setRules(r);
      setHistory(h);
      setTemplates(t);
    } catch (e) {
      console.error("Failed to load alerts:", e);
    }
  };

  useEffect(() => {
    reload();
    const id = setInterval(reload, 10000);
    return () => clearInterval(id);
  }, []);

  const toggleRule = async (rule: AlertRule) => {
    await updateAlertRule(rule.id, { enabled: !rule.enabled });
    reload();
  };

  const removeRule = async (id: string) => {
    if (!confirm("Delete this rule?")) return;
    await deleteAlertRule(id);
    reload();
  };

  const applyTemplate = async (id: string) => {
    if (!confirm(`Apply template "${id}"? This adds its alert rules to the engine.`)) return;
    await applyUseCaseTemplate(id);
    reload();
    setTab("rules");
  };

  return (
    <div style={{ padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
        <button onClick={onBack} style={btn("#374151")}>Back</button>
        <h2 style={{ margin: 0, fontSize: 20 }}>Alert Rules & Templates</h2>
        <span style={{
          marginLeft: "auto",
          padding: "4px 12px",
          background: "#1e293b",
          borderRadius: 12,
          fontSize: 13,
          color: "#94a3b8",
        }}>
          {rules.filter((r) => r.enabled).length} active
        </span>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {[
          { key: "rules" as const, label: "Rules", count: rules.length },
          { key: "history" as const, label: "History", count: history.length },
          { key: "templates" as const, label: "Templates", count: templates.length },
        ].map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{
              padding: "8px 16px",
              background: tab === t.key ? "#3b82f6" : "#1e293b",
              color: tab === t.key ? "#fff" : "#94a3b8",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontWeight: 500,
              fontSize: 13,
            }}
          >
            {t.label} ({t.count})
          </button>
        ))}
      </div>

      {/* Rules tab */}
      {tab === "rules" && (
        <div style={cardStyle}>
          {rules.length === 0 ? (
            <p style={{ color: "#64748b", textAlign: "center", padding: 24 }}>
              No rules configured. Apply a template to get started.
            </p>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {rules.map((r) => (
                <div
                  key={r.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    padding: "12px 16px",
                    background: r.enabled ? "#0f172a" : "#1a1f2e",
                    borderRadius: 6,
                    borderLeft: `4px solid ${SEVERITY_COLORS[r.severity] || "#94a3b8"}`,
                    opacity: r.enabled ? 1 : 0.5,
                  }}
                >
                  <input
                    type="checkbox"
                    checked={r.enabled}
                    onChange={() => toggleRule(r)}
                    style={{ cursor: "pointer", width: 18, height: 18 }}
                  />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 600, color: "#e2e8f0" }}>{r.name}</div>
                    <div style={{ fontSize: 12, color: "#64748b" }}>
                      {r.type} | severity: {r.severity}
                      {r.keywords.length > 0 && ` | keywords: ${r.keywords.join(", ")}`}
                      {r.object_label && ` | object: ${r.object_label}`}
                    </div>
                  </div>
                  <button onClick={() => removeRule(r.id)} style={btn("#dc2626", 12)}>
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* History tab */}
      {tab === "history" && (
        <div style={cardStyle}>
          {history.length === 0 ? (
            <p style={{ color: "#64748b", textAlign: "center", padding: 24 }}>
              No alerts triggered yet.
            </p>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {history.map((h) => (
                <div
                  key={h.id}
                  style={{
                    padding: "10px 14px",
                    background: "#0f172a",
                    borderRadius: 6,
                    borderLeft: `4px solid ${SEVERITY_COLORS[h.severity] || "#94a3b8"}`,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontWeight: 600, color: "#e2e8f0" }}>{h.rule_name}</span>
                    <span style={{ fontSize: 11, color: "#64748b" }}>
                      {new Date(h.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4 }}>
                    Video: {h.video_id.slice(0, 8)} @ {h.triggered_at_sec.toFixed(1)}s — {h.matched_text}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Templates tab */}
      {tab === "templates" && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 12 }}>
          {templates.map((t) => (
            <div key={t.id} style={{ ...cardStyle, padding: 16 }}>
              <h3 style={{ margin: "0 0 6px", fontSize: 16, color: "#e2e8f0" }}>{t.name}</h3>
              <p style={{ margin: "0 0 12px", fontSize: 13, color: "#94a3b8" }}>{t.description}</p>
              <div style={{ fontSize: 12, color: "#64748b", marginBottom: 12 }}>
                <div>{t.alert_rule_count} alert rules</div>
                <div>Chunk size: {t.chunk_sec}s</div>
                <div>Default report: {t.default_report_template}</div>
              </div>
              <button
                onClick={() => applyTemplate(t.id)}
                style={btn("#3b82f6")}
              >
                Apply Template
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const cardStyle: React.CSSProperties = {
  background: "#1e293b",
  borderRadius: 8,
  padding: 16,
};

function btn(bg: string, fontSize = 13): React.CSSProperties {
  return {
    padding: "6px 14px",
    background: bg,
    color: "#fff",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    fontWeight: 600,
    fontSize,
  };
}
