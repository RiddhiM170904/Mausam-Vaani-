function friendlyName(userProfile) {
  const name = String(userProfile?.name || userProfile?.full_name || "").trim();
  return name ? name.split(/\s+/)[0] : "dost";
}

function weatherHint(weather = {}) {
  const temp = Number(weather?.temp || weather?.temperature || 0);
  const rain = Number(weather?.rain_probability || 0);

  if (rain >= 0.5) return "Baarish chance high hai, umbrella le jana.";
  if (temp >= 35) return "Garmi tez hai, pani aur cap carry karo.";
  if (temp <= 12) return "Thandi hawa hai, light jacket useful hogi.";
  return "Weather theek hai, plan normal rakho.";
}

function personaHint(userProfile = {}) {
  const persona = String(userProfile?.persona || userProfile?.user_type || "general").toLowerCase();
  if (persona.includes("driver") || persona.includes("delivery")) {
    return "Road pe nikalne se pehle route check karo.";
  }
  if (persona.includes("student")) {
    return "Nikalne se pehle essentials ready rakho.";
  }
  if (persona.includes("farmer")) {
    return "Field tasks mausam dekh kar plan karo.";
  }
  return "Aaj ka plan mausam ke hisab se rakho.";
}

function buildNotificationPayload({ slotIndex, userProfile, weather }) {
  const name = friendlyName(userProfile);
  const aiSlot = slotIndex % 2 === 1;

  const body = aiSlot
    ? `Hey ${name}, ${personaHint(userProfile)} ${weatherHint(weather)}`
    : `Hey ${name}, weather update: ${weatherHint(weather)}`;

  return {
    source: aiSlot ? "ai" : "reminder",
    title: "Mausam Vaani",
    body,
  };
}

module.exports = { buildNotificationPayload };
