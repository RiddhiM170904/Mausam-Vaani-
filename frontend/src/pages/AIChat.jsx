import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Send, Sparkles } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import GlassCard from "../components/GlassCard";
import Loader from "../components/Loader";

function generateAssistantReply(question, weatherData, persona) {
  const q = question.toLowerCase();
  const current = weatherData?.current || {};
  const temp = Number(current.temp || 0);
  const condition = String(current.condition || "").toLowerCase();
  const wind = Number(current.wind || 0);

  if (q.includes("travel") || q.includes("trip") || q.includes("drive")) {
    if (condition.includes("rain") || wind > 20) {
      return "Travel is possible, but start early, keep buffer time, and avoid low-visibility or waterlogged routes.";
    }
    return "Travel looks reasonably safe. Prefer daytime slots and keep live weather checks enabled.";
  }

  if (q.includes("outside") || q.includes("outdoor")) {
    if (temp >= 35) return "Avoid long outdoor exposure between 12 PM and 4 PM. Use early morning or evening windows.";
    if (condition.includes("rain")) return "Outdoor plans may face interruptions due to rain. Keep an indoor backup.";
    return "Outdoor activity looks fine. Stay hydrated and monitor for quick condition changes.";
  }

  if (q.includes("farmer") || q.includes("irrigation") || q.includes("crop")) {
    if (condition.includes("rain")) return "Delay irrigation for now and re-check after rainfall trend stabilizes.";
    return "Plan irrigation in cooler slots and avoid high-wind windows for spraying tasks.";
  }

  if (q.includes("aqi") || q.includes("pollution") || q.includes("air")) {
    return "If AQI is elevated, reduce strenuous outdoor time and use mask protection during commute windows.";
  }

  if (persona === "driver" || persona === "delivery") {
    return "For road safety, prioritize visibility and rainfall checks before departure. I can also suggest safer time windows if you share your travel time.";
  }

  return "Based on current weather, plan critical tasks in stable windows, keep a backup slot, and use alerts for sudden shifts. Ask me about travel, outdoor safety, or timing.";
}

export default function AIChat() {
  const { user } = useAuth();
  const { location } = useLocation();
  const { data: weatherData, loading } = useWeather(location?.lat, location?.lon);

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Ask me anything about weather-based decisions. Example: Can I travel tomorrow morning?",
    },
  ]);

  const quickPrompts = useMemo(
    () => [
      "Can I travel tomorrow morning?",
      "Is it safe to go outside in afternoon?",
      "Best time for outdoor work today?",
      "Will rain affect delivery schedule?",
    ],
    []
  );

  const sendMessage = async (text) => {
    const question = text.trim();
    if (!question) return;

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setSending(true);

    await new Promise((r) => setTimeout(r, 500));

    const reply = generateAssistantReply(question, weatherData, user?.persona || "general");
    setMessages((prev) => [...prev, { role: "assistant", text: reply }]);
    setSending(false);
  };

  if (loading) return <Loader text="Loading assistant context..." />;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mx-auto w-full max-w-3xl space-y-4 sm:space-y-6">
      <div className="space-y-1 px-1">
        <h1 className="text-2xl font-bold text-white">AI Weather Assistant</h1>
        <p className="text-sm text-gray-400">Conversational support for weather decisions and planning.</p>
      </div>

      <GlassCard className="p-4 sm:p-5" hover={false}>
        <div className="mb-3 flex flex-wrap gap-2">
          {quickPrompts.map((prompt) => (
            <button
              key={prompt}
              onClick={() => sendMessage(prompt)}
              className="rounded-full border border-indigo-500/30 bg-indigo-500/10 px-3 py-1 text-xs text-indigo-300 transition-colors hover:bg-indigo-500/20"
            >
              {prompt}
            </button>
          ))}
        </div>

        <div className="max-h-[55vh] space-y-3 overflow-y-auto pr-1">
          {messages.map((message, idx) => (
            <div key={idx} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${
                  message.role === "user"
                    ? "bg-indigo-500/25 text-indigo-100"
                    : "border border-white/10 bg-white/5 text-gray-200"
                }`}
              >
                {message.role === "assistant" && (
                  <div className="mb-1 flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wide text-indigo-300">
                    <Sparkles className="h-3 w-3" /> AI
                  </div>
                )}
                {message.text}
              </div>
            </div>
          ))}

          {sending && (
            <div className="flex justify-start">
              <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-gray-400">Thinking...</div>
            </div>
          )}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage(input);
          }}
          className="mt-4 flex items-center gap-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a weather decision question..."
            className="w-full rounded-xl border border-gray-700 bg-gray-900/70 px-3 py-2 text-sm text-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button
            type="submit"
            disabled={!input.trim() || sending}
            className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500 text-white transition-colors hover:bg-indigo-600 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </GlassCard>
    </motion.div>
  );
}
