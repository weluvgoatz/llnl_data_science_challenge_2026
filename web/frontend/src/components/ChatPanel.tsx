import { ChevronDown, ChevronUp, LoaderCircle, Send, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ChatTurn, TimelineEntry, ToolCall } from "../types";

const SUBAGENT_NAMES = new Set(["detection_agent", "report_agent", "plot_agent"]);

function ToolCallView({ call }: { call: ToolCall }) {
  return (
    <div className={`tool-call ${call.is_error ? "tool-call-error" : ""}`}>
      <div className="tool-call-name">{call.is_error ? "✕" : "✓"} {call.tool}</div>
      {Object.keys(call.input).length > 0 && (
        <pre className="tool-call-io">{JSON.stringify(call.input, null, 2)}</pre>
      )}
      <pre className="tool-call-io">{JSON.stringify(call.output, null, 2)}</pre>
    </div>
  );
}

function ReasoningTrace({ turn }: { turn: ChatTurn }) {
  const [open, setOpen] = useState(false);
  const totalCalls =
    turn.orchestrator_tool_calls.length +
    Object.values(turn.subagent_traces).reduce(
      (sum, traces) => sum + traces.reduce((inner, trace) => inner + trace.tool_calls.length, 0),
      0,
    );
  if (totalCalls === 0) return null;
  return (
    <div className="reasoning">
      <button className="reasoning-toggle" onClick={() => setOpen((value) => !value)}>
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        {open ? "Hide" : "Show"} reasoning ({totalCalls} tool call{totalCalls === 1 ? "" : "s"})
      </button>
      {open && (
        <div className="reasoning-body">
          {turn.orchestrator_tool_calls.map((call, index) =>
            SUBAGENT_NAMES.has(call.tool) ? (
              <div className="reasoning-step" key={index}>
                <div className="tool-call-name">
                  <Sparkles size={12} /> delegated to {call.tool}
                </div>
                <div className="tool-call-io subagent-request">
                  "{(call.input as { request?: string }).request ?? ""}"
                </div>
                {(turn.subagent_traces[call.tool] || []).map((trace, traceIndex) => (
                  <div className="subagent-trace" key={traceIndex}>
                    {trace.tool_calls.map((sub, subIndex) => (
                      <ToolCallView call={sub} key={subIndex} />
                    ))}
                  </div>
                ))}
              </div>
            ) : (
              <div className="reasoning-step" key={index}>
                <ToolCallView call={call} />
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}

export function ChatPanel({
  timeline,
  onSend,
  busy,
  disabled,
  disabledReason,
}: {
  timeline: TimelineEntry[];
  onSend: (message: string) => void;
  busy: boolean;
  disabled: boolean;
  disabledReason?: string;
}) {
  const [draft, setDraft] = useState("");
  // Optimistic: show the user's own message the instant they send it,
  // rather than waiting for the full round trip to come back as a real
  // ChatTurn. Cleared once the request settles (busy -> false), by which
  // point the real turn has landed in `timeline` and supersedes it.
  const [pending, setPending] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [timeline.length, busy, pending]);

  useEffect(() => {
    if (!busy) setPending(null);
  }, [busy]);

  const submit = () => {
    const message = draft.trim();
    if (!message || busy || disabled) return;
    setPending(message);
    onSend(message);
    setDraft("");
  };

  return (
    <aside className="chat-pane">
      <div className="chat-messages" ref={scrollRef}>
        {timeline.length === 0 && (
          <div className="chat-empty">
            <p>
              Lattice Lens compares an additively-manufactured lattice part&rsquo;s intended design against its CT
              scan to find missing or defective struts.
            </p>
            <ul className="chat-empty-files">
              <li>
                <strong>STL</strong> &mdash; intended CAD geometry
              </li>
              <li>
                <strong>TIFF</strong> &mdash; CT scan of the printed part
              </li>
              <li>
                <strong>JSON</strong> &mdash; lattice graph linking design to individual struts
              </li>
            </ul>
            <p>
              Following upload, query the system on your structure&rsquo;s geometry, scan, and inspection results
              &mdash; &ldquo;show the model,&rdquo; &ldquo;show the TIFF,&rdquo; &ldquo;run the analysis,&rdquo; and
              ask about defects once it&rsquo;s done.
            </p>
          </div>
        )}
        {timeline.map((entry) => {
          if (entry.kind === "status") {
            return (
              <div className="status-event" key={entry.id}>
                {entry.text}
              </div>
            );
          }
          if (entry.kind === "announcement") {
            // Same visual weight as a real assistant reply (per the "the
            // agent must post a clear completion message" requirement) --
            // still frontend-synthesized from real polled state, not a new
            // model call, but deliberately not distinguished from a turn
            // visually so the user reliably notices it.
            return (
              <div className="chat-bubble assistant announcement" key={entry.id}>
                <p>{entry.text}</p>
              </div>
            );
          }
          return (
            <div className="chat-turn" key={entry.at}>
              <div className="chat-bubble user">{entry.turn.user_message}</div>
              <div className="chat-bubble assistant">
                <p>{entry.turn.reply}</p>
                <ReasoningTrace turn={entry.turn} />
              </div>
            </div>
          );
        })}
        {pending && <div className="chat-bubble user">{pending}</div>}
        {busy && (
          <div className="chat-bubble assistant pending">
            <LoaderCircle className="spin" size={14} /> Thinking…
          </div>
        )}
      </div>
      <div className="chat-composer">
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          placeholder={disabled ? disabledReason || "Chat is unavailable" : "Ask a question…"}
          disabled={disabled}
          rows={2}
        />
        <button
          className="icon-button chat-send"
          onClick={submit}
          disabled={disabled || busy || !draft.trim()}
          aria-label="Send message"
        >
          <Send size={16} />
        </button>
      </div>
    </aside>
  );
}
