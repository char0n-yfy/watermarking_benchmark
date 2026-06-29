"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

interface ResourcePaginationProps {
  page: number;
  pageCount: number;
  onPageChange: (page: number) => void;
  previousLabel: string;
  nextLabel: string;
}

function visiblePages(page: number, pageCount: number): number[] {
  if (pageCount <= 9) {
    return Array.from({ length: pageCount }, (_, index) => index + 1);
  }

  const pages = new Set<number>([1, pageCount, page - 1, page, page + 1]);
  return [...pages].filter((value) => value >= 1 && value <= pageCount).sort((left, right) => left - right);
}

export function ResourcePagination({
  page,
  pageCount,
  onPageChange,
  previousLabel,
  nextLabel
}: ResourcePaginationProps) {
  if (pageCount <= 1) {
    return null;
  }

  const pages = visiblePages(page, pageCount);

  return (
    <nav aria-label="Pagination" className="pagination-row">
      <button
        className="icon-button"
        disabled={page <= 1}
        onClick={() => onPageChange(page - 1)}
        title={previousLabel}
        type="button"
      >
        <ChevronLeft size={16} />
      </button>

      <div className="pagination-pages">
        {pages.map((pageNumber, index) => {
          const previous = pages[index - 1];
          const showEllipsis = previous != null && pageNumber - previous > 1;
          return (
            <span className="pagination-page-slot" key={pageNumber}>
              {showEllipsis ? <span className="pagination-ellipsis">…</span> : null}
              <button
                aria-current={pageNumber === page ? "page" : undefined}
                className={pageNumber === page ? "pagination-page-button active" : "pagination-page-button"}
                onClick={() => onPageChange(pageNumber)}
                type="button"
              >
                {pageNumber}
              </button>
            </span>
          );
        })}
      </div>

      <span className="pagination-summary">
        {page} / {pageCount}
      </span>

      <button
        className="icon-button"
        disabled={page >= pageCount}
        onClick={() => onPageChange(page + 1)}
        title={nextLabel}
        type="button"
      >
        <ChevronRight size={16} />
      </button>
    </nav>
  );
}
