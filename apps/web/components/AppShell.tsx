"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";
import {
  Activity,
  Boxes,
  Database,
  Images,
  Languages,
  LayoutDashboard,
  Menu,
  SlidersHorizontal,
  Trophy,
  X
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
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const mobileNavLabel = language === "zh" ? "主导航" : "Main navigation";

  return (
    <div className="shell">
      <a className="skip-link" href="#main-content">
        {language === "zh" ? "跳到主内容" : "Skip to main content"}
      </a>
      <aside className={mobileNavOpen ? "sidebar mobile-open" : "sidebar"}>
        <div className="sidebar-header">
          <Link aria-label={t.console.projectName} className="brand" href="/" onClick={() => setMobileNavOpen(false)}>
            <div className="brand-mark">
              <img alt="" aria-hidden="true" className="brand-logo" src="/waterprism-logo.png" />
            </div>
            <span>WaterPrism</span>
          </Link>
          <button
            aria-controls="primary-navigation-panel"
            aria-expanded={mobileNavOpen}
            aria-label={mobileNavOpen ? (language === "zh" ? "关闭主导航" : "Close main navigation") : (language === "zh" ? "打开主导航" : "Open main navigation")}
            className="mobile-nav-toggle"
            onClick={() => setMobileNavOpen((current) => !current)}
            type="button"
          >
            {mobileNavOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
        <div className="sidebar-panel" id="primary-navigation-panel">
          <nav aria-label={mobileNavLabel} className="nav">
            {nav.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  aria-current={active === item.key ? "page" : undefined}
                  className={`nav-link ${active === item.key ? "active" : ""}`}
                  href={item.href}
                  key={item.href}
                  onClick={() => setMobileNavOpen(false)}
                >
                  <Icon size={16} />
                  {t.nav[item.key]}
                </Link>
              );
            })}
          </nav>
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
        </div>
      </aside>
      <main className="main" id="main-content" tabIndex={-1}>{children}</main>
    </div>
  );
}
