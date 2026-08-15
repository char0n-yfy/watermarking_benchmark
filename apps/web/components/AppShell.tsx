"use client";

import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";
import {
  Activity,
  Boxes,
  ChevronLeft,
  ChevronRight,
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
  { href: "/leaderboard", key: "leaderboard", icon: Trophy }
] as const;

type ActiveNav = (typeof nav)[number]["key"];

export function AppShell({ active, children }: { active: ActiveNav; children: ReactNode }) {
  const { language, setLanguage, t } = useLanguage();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const mobileNavLabel = language === "zh" ? "主导航" : "Main navigation";
  const collapseLabel = language === "zh" ? "收起侧边栏" : "Collapse sidebar";
  const expandLabel = language === "zh" ? "展开侧边栏" : "Expand sidebar";

  useEffect(() => {
    try {
      const collapsed = window.localStorage.getItem("waterprism-sidebar-collapsed") === "true";
      setSidebarCollapsed(collapsed);
      if (collapsed) {
        document.documentElement.dataset.sidebarCollapsed = "true";
      } else {
        delete document.documentElement.dataset.sidebarCollapsed;
      }
    } catch {
      setSidebarCollapsed(false);
      delete document.documentElement.dataset.sidebarCollapsed;
    }
  }, []);

  useEffect(() => {
    const title = `${t.nav[active]} · WaterPrism`;
    const syncDocumentMetadata = () => {
      if (document.title !== title) {
        document.title = title;
      }
      document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    };
    syncDocumentMetadata();
    const observer = new MutationObserver(syncDocumentMetadata);
    observer.observe(document.head, { characterData: true, childList: true, subtree: true });
    return () => observer.disconnect();
  }, [active, language, t.nav]);

  const toggleSidebar = () => {
    setSidebarCollapsed((current) => {
      const next = !current;
      try {
        window.localStorage.setItem("waterprism-sidebar-collapsed", String(next));
      } catch {
        // Keep the control functional when browser storage is unavailable.
      }
      if (next) {
        document.documentElement.dataset.sidebarCollapsed = "true";
      } else {
        delete document.documentElement.dataset.sidebarCollapsed;
      }
      return next;
    });
  };

  return (
    <div className={sidebarCollapsed ? "shell sidebar-collapsed" : "shell"}>
      <a className="skip-link" href="#main-content">
        {language === "zh" ? "跳到主内容" : "Skip to main content"}
      </a>
      <aside className={mobileNavOpen ? "sidebar mobile-open" : "sidebar"}>
        <div className="sidebar-header">
          <Link aria-label={t.console.projectName} className="brand" href="/" onClick={() => setMobileNavOpen(false)}>
            <div className="brand-mark">
              <img alt="" aria-hidden="true" className="brand-logo" src="/waterprism-logo.png" />
            </div>
            <span className="brand-label">WaterPrism</span>
          </Link>
          <button
            aria-label={sidebarCollapsed ? expandLabel : collapseLabel}
            className="desktop-sidebar-toggle"
            onClick={toggleSidebar}
            type="button"
          >
            {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
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
                  aria-label={sidebarCollapsed ? t.nav[item.key] : undefined}
                >
                  <Icon size={16} />
                  <span className="nav-label">{t.nav[item.key]}</span>
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
