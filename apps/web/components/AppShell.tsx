"use client";

import type { ReactNode } from "react";
import {
  Activity,
  Boxes,
  Database,
  FlaskConical,
  Images,
  Languages,
  LayoutDashboard,
  SlidersHorizontal,
  Trophy
} from "lucide-react";
import { useLanguage } from "@/components/LanguageProvider";
import { languages } from "@/lib/i18n";

const nav = [
  { href: "/", key: "console", icon: LayoutDashboard },
  { href: "/configs", key: "configs", icon: SlidersHorizontal },
  { href: "/resources", key: "resources", icon: Boxes },
  { href: "/runs", key: "runs", icon: Activity },
  { href: "/results", key: "results", icon: Images },
  { href: "/leaderboard", key: "leaderboard", icon: Trophy },
  { href: "/schema", key: "schema", icon: Database }
] as const;

type ActiveNav = (typeof nav)[number]["key"];

export function AppShell({ active, children }: { active: ActiveNav; children: ReactNode }) {
  const { language, setLanguage, t } = useLanguage();

  return (
    <div className="shell">
      <aside className="sidebar">
        <div>
          <div className="brand">
            <div className="brand-mark">
              <FlaskConical size={17} />
            </div>
            <span>WM Bench</span>
          </div>
          <nav className="nav">
            {nav.map((item) => {
              const Icon = item.icon;
              return (
                <a
                  className={`nav-link ${active === item.key ? "active" : ""}`}
                  href={item.href}
                  key={item.href}
                >
                  <Icon size={16} />
                  {t.nav[item.key]}
                </a>
              );
            })}
          </nav>
        </div>
        <div className="language-panel">
          <div className="language-label">
            <Languages size={14} />
            <span>{t.languageLabel}</span>
          </div>
          <div className="language-toggle" role="group" aria-label={t.languageLabel}>
            {languages.map((item) => (
              <button
                aria-pressed={language === item.code}
                className={language === item.code ? "active" : ""}
                key={item.code}
                onClick={() => setLanguage(item.code)}
                type="button"
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
