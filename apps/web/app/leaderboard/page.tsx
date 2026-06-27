"use client";

import { BarChart3, CheckCircle2, Gauge, ImageIcon, ShieldCheck, Timer, Trophy } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";

export default function LeaderboardPage() {
  const { language, t } = useLanguage();
  const dimensions = [
    { label: t.leaderboard.robustness, icon: ShieldCheck, weight: "0.50" },
    { label: t.leaderboard.imageQuality, icon: ImageIcon, weight: "0.30" },
    { label: t.leaderboard.speed, icon: Timer, weight: "0.20" },
    { label: t.leaderboard.overall, icon: Trophy, weight: "1.00" }
  ];
  const requirements =
    language === "zh"
      ? ["数据集", "攻击预设", "强度网格", "随机种子", "最小样本数", "评分公式"]
      : ["Datasets", "Attack presets", "Strength grid", "Seeds", "Minimum samples", "Scoring formula"];

  return (
    <AppShell active="leaderboard">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.leaderboard.title}</h1>
          <p>{t.leaderboard.subtitle}</p>
        </div>
      </div>

      <section className="leaderboard-grid">
        <div className="panel leaderboard-hero">
          <div className="leaderboard-empty">
            <div className="leaderboard-icon">
              <Trophy size={26} />
            </div>
            <div>
              <h2>{t.leaderboard.pendingTitle}</h2>
              <p>{t.leaderboard.pendingBody}</p>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.leaderboard.metrics}</h2>
            <BarChart3 size={16} />
          </div>
          <div className="panel-body metric-list">
            {dimensions.map((dimension) => {
              const Icon = dimension.icon;
              return (
                <div className="metric-row" key={dimension.label}>
                  <div>
                    <Icon size={16} />
                    <span>{dimension.label}</span>
                  </div>
                  <strong>{dimension.weight}</strong>
                </div>
              );
            })}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.leaderboard.requirements}</h2>
            <Gauge size={16} />
          </div>
          <div className="panel-body requirement-grid">
            {requirements.map((item) => (
              <div className="requirement-item" key={item}>
                <CheckCircle2 size={15} />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </AppShell>
  );
}
