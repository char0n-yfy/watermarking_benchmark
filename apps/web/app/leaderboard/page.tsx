"use client";

import { useEffect, useMemo, useState } from "react";
import { BarChart3, CheckCircle2, Gauge, ShieldCheck, Trophy } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { BenchmarkRadar } from "@/components/BenchmarkRadar";
import { useLanguage } from "@/components/LanguageProvider";
import { fetchBenchmarkProtocols, fetchLeaderboard } from "@/lib/api";
import { formatMetric } from "@/lib/insights";
import type { BenchmarkProtocol, LeaderboardResponse } from "@/lib/types";

export default function LeaderboardPage() {
  const { language, t } = useLanguage();
  const [protocols, setProtocols] = useState<BenchmarkProtocol[]>([]);
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchBenchmarkProtocols(), fetchLeaderboard()])
      .then(([nextProtocols, nextLeaderboard]) => {
        if (cancelled) {
          return;
        }
        setProtocols(nextProtocols);
        setLeaderboard(nextLeaderboard);
      })
      .catch(() => {
        if (!cancelled) {
          setNotice(language === "zh" ? "API 未启动或暂无评分数据。" : "API is unavailable or no scores exist yet.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [language]);

  const rows = leaderboard?.rows ?? [];
  const topRow = rows[0] ?? null;
  const protocol = leaderboard?.protocol ?? protocols[0] ?? null;
  const requirements =
    language === "zh"
      ? ["固定 5000 张样本", "正负样本同攻击流程", "七类 WAVES 攻击", "TPR@0.1%FPR", "NQD < 0.8", "完整 coverage"]
      : ["Fixed 5,000 samples", "Matched positive/negative attacks", "Seven WAVES attack classes", "TPR@0.1%FPR", "NQD < 0.8", "Full coverage"];

  const officialRows = useMemo(() => rows.filter((row) => row.officialEligible), [rows]);
  const provisionalRows = useMemo(() => rows.filter((row) => !row.officialEligible), [rows]);

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
          {rows.length === 0 ? (
            <div className="leaderboard-empty">
              <div className="leaderboard-icon">
                <Trophy size={26} />
              </div>
              <div>
                <h2>{t.leaderboard.pendingTitle}</h2>
                <p>{notice || t.leaderboard.noRows}</p>
              </div>
            </div>
          ) : (
            <>
              <div className="panel-header">
                <h2>{topRow?.officialEligible ? t.leaderboard.officialRows : t.leaderboard.provisionalRows}</h2>
                <span className={topRow?.officialEligible ? "badge ok" : "badge warn"}>
                  {topRow?.officialEligible ? t.common.official : t.common.provisional}
                </span>
              </div>
              <div className="panel-body table-scroll">
                <ScoreTable rows={officialRows.length ? officialRows : provisionalRows} />
              </div>
            </>
          )}
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.results.radar}</h2>
            <BarChart3 size={16} />
          </div>
          <div className="panel-body">
            <BenchmarkRadar categories={topRow?.categoryScores ?? []} emptyText={t.leaderboard.pendingBody} />
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.leaderboard.protocol}</h2>
            <ShieldCheck size={16} />
          </div>
          <div className="panel-body metric-list">
            <div className="metric-row">
              <div>
                <Gauge size={16} />
                <span>{protocol?.name ?? "waves-official-detection-v1"}</span>
              </div>
              <strong>{protocol?.task ?? "detection"}</strong>
            </div>
            <div className="metric-row">
              <span>{t.results.tprAtFpr}</span>
              <strong>{protocol ? `${(protocol.fprTarget * 100).toFixed(1)}% FPR` : "0.1% FPR"}</strong>
            </div>
            <div className="metric-row">
              <span>{t.results.nqd}</span>
              <strong>&lt; {protocol?.practicalNqdThreshold ?? 0.8}</strong>
            </div>
            <div className="metric-row">
              <span>{t.common.samples}</span>
              <strong>{protocol?.officialMinSamples ?? 5000}</strong>
            </div>
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

  function ScoreTable({ rows: tableRows }: { rows: LeaderboardResponse["rows"] }) {
    if (tableRows.length === 0) {
      return <div className="empty compact-empty">{t.leaderboard.noRows}</div>;
    }
    return (
      <table className="table compact-table">
        <thead>
          <tr>
            <th>{t.common.rank}</th>
            <th>{t.common.algorithm}</th>
            <th>{t.common.wrs}</th>
            <th>{t.results.cleanFidelity}</th>
            <th>{t.results.nqd}</th>
            <th>{t.common.coverage}</th>
            <th>{t.common.runtime}</th>
          </tr>
        </thead>
        <tbody>
          {tableRows.map((row) => (
            <tr key={`${row.runId ?? "run"}-${row.algorithmId}`}>
              <td>{row.rank}</td>
              <td>
                <strong>{row.algorithmId}</strong>
                <span className="subtle-cell">{row.configName ?? row.runId}</span>
              </td>
              <td>{row.wrs == null ? "n/a" : row.wrs.toFixed(1)}</td>
              <td>{formatMetric(row.cleanFidelity)}</td>
              <td>{formatMetric(row.avgNqd)}</td>
              <td>
                {row.coverage.coveredCategoryCount}/{row.coverage.requiredCategoryCount}
              </td>
              <td>{row.runtimeMs == null ? "n/a" : `${(row.runtimeMs / 1000).toFixed(2)}s`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
}
