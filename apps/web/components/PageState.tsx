import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

export type PageStateTone = "loading" | "empty" | "error";

export function PageState({
  actions,
  description,
  icon: Icon,
  title,
  tone = "empty"
}: {
  actions?: ReactNode;
  description: string;
  icon: LucideIcon;
  title: string;
  tone?: PageStateTone;
}) {
  return (
    <section
      aria-live={tone === "loading" ? "polite" : undefined}
      className={`page-state ${tone}`}
      role={tone === "error" ? "alert" : "status"}
    >
      <span aria-hidden="true" className="page-state-icon">
        <Icon className={tone === "loading" ? "loading-spinner" : undefined} size={26} />
      </span>
      <div className="page-state-copy">
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-state-actions">{actions}</div> : null}
    </section>
  );
}
