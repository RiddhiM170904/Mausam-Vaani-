import { AlertTriangle } from "lucide-react";

export default function ServiceUnavailable({
  title = "Service Temporarily Unavailable",
  message = "We are not servicing at this time. Sorry for the inconvenience.",
  showRetry = false,
  onRetry,
}) {
  return (
    <div className="flex min-h-[60vh] items-center justify-center px-4">
      <div className="w-full max-w-md rounded-2xl border border-red-500/25 bg-red-500/10 p-6 text-center backdrop-blur-xl">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-red-500/20">
          <AlertTriangle className="h-6 w-6 text-red-300" />
        </div>
        <h1 className="text-lg font-semibold text-white">{title}</h1>
        <p className="mt-2 text-sm text-red-100/90">{message}</p>

        {showRetry && (
          <button
            onClick={onRetry}
            className="mt-5 rounded-xl border border-red-400/40 bg-red-500/20 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-500/30"
          >
            Try Again
          </button>
        )}
      </div>
    </div>
  );
}
