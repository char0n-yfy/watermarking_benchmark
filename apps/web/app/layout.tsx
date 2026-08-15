import type { Metadata } from "next";
import type { ReactNode } from "react";
import { LanguageProvider } from "@/components/LanguageProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "Watermark Benchmark",
  description: "Research console for watermark robustness experiments"
};

const sidebarPreferenceScript = `
  try {
    if (window.localStorage.getItem("waterprism-sidebar-collapsed") === "true") {
      document.documentElement.dataset.sidebarCollapsed = "true";
    }
  } catch {}
`;

export default function RootLayout({
  children
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: sidebarPreferenceScript }} />
      </head>
      <body>
        <LanguageProvider>{children}</LanguageProvider>
      </body>
    </html>
  );
}
