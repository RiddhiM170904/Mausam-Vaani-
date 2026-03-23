const fs = require('fs');
const path = require('path');

const rulesPath = path.join(__dirname, '../data/rag_rules.json');

function loadRules() {
  try {
    const raw = fs.readFileSync(rulesPath, 'utf-8');
    return JSON.parse(raw);
  } catch (error) {
    console.error('RAG rules load failed:', error.message);
    return [];
  }
}

function compareRange(value, range) {
  if (typeof value !== 'number' || !range) {
    return false;
  }

  if (typeof range.gte === 'number' && value < range.gte) {
    return false;
  }

  if (typeof range.gt === 'number' && value <= range.gt) {
    return false;
  }

  if (typeof range.lte === 'number' && value > range.lte) {
    return false;
  }

  if (typeof range.lt === 'number' && value >= range.lt) {
    return false;
  }

  return true;
}

function intersectsCaseInsensitive(values = [], allowed = []) {
  if (!Array.isArray(values) || !Array.isArray(allowed) || !allowed.length) {
    return true;
  }

  const left = values.map((item) => String(item).toLowerCase());
  const right = new Set(allowed.map((item) => String(item).toLowerCase()));

  return left.some((item) => right.has(item));
}

function ruleMatches(rule, context) {
  const when = rule.when || {};
  const weather = context.weather || {};
  const user = context.user_profile || context.userProfile || {};
  const userType = user.user_type;
  const vehicle = user.profile?.vehicle;

  if (when.temp && !compareRange(weather.temp, when.temp)) {
    return false;
  }

  if (when.rain_probability && !compareRange(weather.rain_probability, when.rain_probability)) {
    return false;
  }

  if (when.aqi && !compareRange(weather.aqi, when.aqi)) {
    return false;
  }

  if (when.wind_speed && !compareRange(weather.wind_speed, when.wind_speed)) {
    return false;
  }

  if (when.humidity && !compareRange(weather.humidity, when.humidity)) {
    return false;
  }

  if (when.user_types && !intersectsCaseInsensitive([userType], when.user_types)) {
    return false;
  }

  if (when.vehicles && !intersectsCaseInsensitive([vehicle], when.vehicles)) {
    return false;
  }

  return true;
}

function retrieveKnowledge(context, limit = 6) {
  const rules = loadRules();

  const matched = rules
    .filter((rule) => ruleMatches(rule, context))
    .sort((a, b) => {
      const severityRank = { high: 3, medium: 2, low: 1 };
      const aRank = severityRank[a.severity] || 0;
      const bRank = severityRank[b.severity] || 0;
      return bRank - aRank;
    })
    .slice(0, limit);

  return matched.map((rule) => ({
    id: rule.id,
    type: rule.type,
    severity: rule.severity,
    text: rule.advice,
  }));
}

module.exports = {
  retrieveKnowledge,
};
