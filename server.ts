import express, { Request, Response, NextFunction } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import jwt from 'jsonwebtoken';
import { GoogleGenAI } from '@google/genai';

import {
  calculateLandAreaSqm,
  calculateWaterBudget,
  generateRecommendations,
  planWaterDistribution,
  evaluateViability,
  determineDroughtMode
} from './src/services/waterEngine.ts';
import { fetchWeatherForDistrict } from './src/services/weatherService.ts';
import { db } from './src/server/database.ts';
import {
  FarmerInputRequest,
  FullPlannerResponse,
  FarmerDashboardData,
  SupportedLanguage
} from './src/types/index.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const JWT_SECRET = process.env.JWT_SECRET || 'waterwise-farmer-secure-secret-token-2026';

app.use(express.json());

// Extend Express Request for authenticated farmer
interface AuthenticatedRequest extends Request {
  farmerId?: string;
  farmerMobile?: string;
}

// Authentication Middleware
function authenticateFarmer(req: AuthenticatedRequest, res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized: Authentication token required' });
  }

  const token = authHeader.split(' ')[1];
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as { farmer_id: string; mobile: string };
    const farmer = db.findFarmerById(decoded.farmer_id);
    if (!farmer) {
      return res.status(401).json({ error: 'Unauthorized: Farmer account not found' });
    }
    req.farmerId = decoded.farmer_id;
    req.farmerMobile = decoded.mobile;
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Unauthorized: Invalid or expired session' });
  }
}

// Initialize Google GenAI if key is present
let aiClient: GoogleGenAI | null = null;
if (process.env.GEMINI_API_KEY) {
  try {
    aiClient = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  } catch (err) {
    console.error('Failed to initialize GoogleGenAI client:', err);
  }
}

// Cooldown management for Gemini API rate limits & free-tier quota exhaustion
let geminiCooldownUntil = 0;

function isGeminiInCooldown(): boolean {
  return Date.now() < geminiCooldownUntil;
}

function handleGeminiError(err: any, context: string): void {
  const errMsg = typeof err === 'string' ? err : err?.message || JSON.stringify(err || '');
  const is429 = errMsg.includes('429') || errMsg.includes('RESOURCE_EXHAUSTED') || errMsg.includes('quota') || errMsg.includes('Quota exceeded');

  if (is429) {
    let cooldownMs = 30 * 60 * 1000; // default 30 minutes
    const matchSeconds = errMsg.match(/retryDelay"?\s*:\s*"?(\d+)s/i);
    const matchMinutes = errMsg.match(/retry in (\d+)m/i);
    if (matchSeconds) {
      cooldownMs = (parseInt(matchSeconds[1], 10) + 15) * 1000;
    } else if (matchMinutes) {
      cooldownMs = (parseInt(matchMinutes[1], 10) + 1) * 60 * 1000;
    }
    geminiCooldownUntil = Date.now() + cooldownMs;
    console.log(`[WaterWise] Daily API quota reached on free tier. Switched to Karnataka Agronomic Dryland Engine for ${Math.ceil(cooldownMs / 60000)}m.`);
  } else {
    console.log(`[WaterWise] Notice: Local agronomic dryland model engaged for ${context}.`);
  }
}

// In-memory cache for agronomic insights
const agronomyInsightsCache = new Map<string, { insights: string[]; isAiPowered: boolean }>();

// Generate rule-based high-value dryland field directives grounded in official benchmarks
function generateRuleBasedDrylandInsights(
  input: FarmerInputRequest,
  deficitOrSurplus: number,
  reserve: number
): string[] {
  const cropLower = input.crop.toLowerCase();
  const soilLower = input.soil_type.toLowerCase();
  const directives: string[] = [];

  // Directive 1: Soil texture and evaporation mitigation (M03 / M12)
  if (soilLower.includes('black')) {
    directives.push(`Deep Vertisols in ${input.district} develop shrink-cracks; apply shallow intercultural weeding or straw mulch (M03) to seal cracks and conserve 20% soil moisture.`);
  } else if (soilLower.includes('red') || soilLower.includes('sandy')) {
    directives.push(`Red/sandy soils have high percolation; deliver short, frequent pulse cycles (M01/M02) rather than long flooding to keep water in the active root zone.`);
  } else {
    directives.push(`In ${input.district}'s ${input.soil_type}, nocturnal or twilight drip watering (after 6:30 PM) cuts midday evaporative losses by up to 25%.`);
  }

  // Directive 2: Critical growth stage water locking (M06 / P01)
  if (cropLower.includes('cotton')) {
    directives.push(`For Cotton, lock at least 50% of available reserve for flowering to boll development; alternate furrow irrigation (M18) prevents boll drop.`);
  } else if (cropLower.includes('maize')) {
    directives.push(`For Maize, silking and tasseling are hyper-critical to drought; protect moisture at flowering to prevent barren cobs and 40% yield drop.`);
  } else if (cropLower.includes('jowar') || cropLower.includes('sorghum')) {
    directives.push(`Jowar is drought-hardy; provide life-saving protective watering primarily during boot leaf and grain fill stages (M06).`);
  } else if (cropLower.includes('groundnut')) {
    directives.push(`Groundnut requires moisture at peg penetration and pod development (M06); micro-sprinklers prevent peg desiccation in dry soil.`);
  } else if (cropLower.includes('rice') || cropLower.includes('paddy')) {
    directives.push(`Adopt Alternate Wetting and Drying (M07) or Direct Seeded Rice (M08) to reduce irrigation water volume by 25-30%.`);
  } else if (cropLower.includes('tur')) {
    directives.push(`Tur (Pigeonpea) deep taproot extracts subsoil water; provide 1-2 protective waterings at flowering and pod fill stages.`);
  } else {
    directives.push(`For ${input.crop}, prioritize dedicating at least 45% of your available reserve to the critical flowering window to secure yield.`);
  }

  // Directive 3: Reserve & Deficit management (P01 / P02 / P03)
  if (deficitOrSurplus < 0 && Math.abs(deficitOrSurplus) > reserve) {
    directives.push(`Severe water deficit: thin the plant population by 25-30% (P02) and spray 1% KNO3 or 5% Kaolin to cut transpiration.`);
  } else if (deficitOrSurplus < 0) {
    directives.push(`Moderate deficit: adopt alternate-row deficit irrigation (M18) to stretch reserve across the entire productive season.`);
  } else {
    directives.push(`Healthy reserve: schedule fertigation through drip system (M01) during active vegetative growth to maximize fertilizer uptake.`);
  }

  // Directive 4: Official Schemes & Weather scheduling (M05 / S01 / PMKSY)
  directives.push(`Apply for 70% to 90% micro-irrigation subsidy under the Karnataka PMKSY Per Drop More Crop scheme via the DBT Raitha Mitra portal.`);

  return directives.slice(0, 4);
}

// Generate expert agronomic dryland insights using Gemini or rule-based fallback
async function getAgronomicInsights(
  input: FarmerInputRequest,
  deficitOrSurplus: number,
  seasonalNeed: number,
  reserve: number
): Promise<{ insights: string[]; isAiPowered: boolean }> {
  const cacheKey = `${input.district}_${input.crop}_${input.soil_type}_${input.growth_stage}_${input.weather_scenario || 'normal'}_${Math.round(deficitOrSurplus / 25000)}`;

  if (agronomyInsightsCache.has(cacheKey)) {
    return agronomyInsightsCache.get(cacheKey)!;
  }

  const fallbackInsights = generateRuleBasedDrylandInsights(input, deficitOrSurplus, reserve);

  // If Gemini is currently in cooldown due to quota limit or client is missing, use rule-based immediately
  if (isGeminiInCooldown() || !aiClient || !process.env.GEMINI_API_KEY) {
    const result = { insights: fallbackInsights, isAiPowered: false };
    agronomyInsightsCache.set(cacheKey, result);
    return result;
  }

  try {
    const prompt = `You are a senior dryland agronomist and water-management specialist for Karnataka, India.
A farmer has provided the following profile:
- District: ${input.district} (North Karnataka)
- Crop: ${input.crop}
- Land size: ${input.land_size} ${input.land_unit || 'acres'}
- Soil texture: ${input.soil_type}
- Water source: ${input.water_source}
- Available reserve: ${reserve.toLocaleString('en-IN')} Litres
- Calculated seasonal need: ${seasonalNeed.toLocaleString('en-IN')} Litres
- Water deficit/surplus: ${deficitOrSurplus.toLocaleString('en-IN')} Litres
- Current growth stage: ${input.growth_stage}

Provide exactly 3 or 4 concise, highly practical bullet points with actionable dryland water-saving advice.
Keep each bullet point under 25 words. Do not include markdown bold or greeting pleasantries.`;

    const generatePromise = aiClient.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
    });

    const timeoutPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('Gemini API timeout')), 3500)
    );

    const response = await Promise.race([generatePromise, timeoutPromise]);

    const text = response.text || '';
    const lines = text
      .split('\n')
      .map(line => line.replace(/^[\s-*•\d.]+/, '').trim())
      .filter(line => line.length > 10);

    if (lines.length >= 2) {
      const result = { insights: lines.slice(0, 4), isAiPowered: true };
      agronomyInsightsCache.set(cacheKey, result);
      return result;
    }
  } catch (err) {
    handleGeminiError(err, 'agronomic advisory');
  }

  const result = { insights: fallbackInsights, isAiPowered: false };
  agronomyInsightsCache.set(cacheKey, result);
  return result;
}

// ----------------------------------------------------
// Public API Routes
// ----------------------------------------------------
app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'WaterWise – Smart Water-Saver Planner & Farmer Portal API',
    region: 'Karnataka Dry Zones',
    ai_available: Boolean(aiClient && process.env.GEMINI_API_KEY && !isGeminiInCooldown()),
    cooldown_active: isGeminiInCooldown(),
  });
});

app.get('/api/weather/:district', async (req, res) => {
  try {
    const district = req.params.district;
    const scenario = (req.query.scenario as any) || 'normal';
    const landSqm = 4046.86;
    const weather = await fetchWeatherForDistrict(district, landSqm, scenario);
    res.json(weather);
  } catch (error) {
    res.status(500).json({ error: 'Failed to retrieve weather data' });
  }
});

app.post('/api/planner/generate-plan', async (req, res) => {
  try {
    const input: FarmerInputRequest = req.body;

    if (!input.crop || !input.district || !input.land_size || !input.soil_type) {
      return res.status(400).json({ error: 'Missing mandatory fields (crop, district, land_size, soil_type)' });
    }

    const landAreaSqm = calculateLandAreaSqm(input.land_size, input.land_unit);
    const weather = await fetchWeatherForDistrict(input.district, landAreaSqm, input.weather_scenario || 'normal');
    const calculations = calculateWaterBudget(input, weather.effective_rainfall_liters);
    const recommendations = generateRecommendations(input, calculations);

    let waterAllocation = null;
    if (calculations.available_reserve_liters > 0) {
      waterAllocation = planWaterDistribution(input.crop, calculations.available_reserve_liters);
    }

    const viability = evaluateViability(input, calculations);
    const droughtMode = determineDroughtMode(input, calculations, weather.weekly_rainfall_total_mm);

    const { insights, isAiPowered } = await getAgronomicInsights(
      input,
      calculations.deficit_or_surplus_liters,
      calculations.seasonal_water_need_liters,
      calculations.available_reserve_liters
    );

    const fullResponse: FullPlannerResponse = {
      farmer_profile: input,
      weather,
      calculations,
      recommendations,
      water_allocation: waterAllocation,
      drought_mode: droughtMode,
      viability,
      ai_summary_insights: insights,
      is_ai_powered: isAiPowered,
    };

    res.json(fullResponse);
  } catch (error) {
    console.error('Error generating plan:', error);
    res.status(500).json({ error: 'Internal server error processing agricultural plan' });
  }
});

// ----------------------------------------------------
// Farmer Authentication Routes
// ----------------------------------------------------
app.get('/api/auth/demo-farmers', (_req, res) => {
  const demos = db.getAllDemoFarmers();
  res.json({
    success: true,
    farmers: demos,
    demo_password: 'karnataka123'
  });
});

app.post('/api/auth/send-otp', (req, res) => {
  const { mobile } = req.body;
  if (!mobile || !/^\d{10}$/.test(String(mobile).trim())) {
    return res.status(400).json({ error: 'Please enter a valid 10-digit mobile number' });
  }

  // Generate 6-digit OTP
  const code = String(Math.floor(100000 + Math.random() * 900000));
  db.saveOtp(mobile, code, 5);

  const farmer = db.findFarmerByMobile(mobile);

  res.json({
    success: true,
    message: `OTP sent successfully to +91-${mobile}`,
    is_registered: Boolean(farmer),
    farmer_name: farmer ? farmer.name : null,
    // Demo environment transparent OTP delivery
    demo_otp: code
  });
});

app.post('/api/auth/verify-otp', (req, res) => {
  const { mobile, otp } = req.body;
  if (!mobile || !otp) {
    return res.status(400).json({ error: 'Mobile number and OTP are required' });
  }

  const result = db.verifyOtp(mobile, otp);
  if (!result.valid) {
    return res.status(400).json({ error: result.message });
  }

  res.json({ success: true, verified: true, message: result.message });
});

app.post('/api/auth/login', (req, res) => {
  const { mobile, password } = req.body;
  if (!mobile || !password) {
    return res.status(400).json({ error: 'Mobile number and password are required' });
  }

  const farmer = db.findFarmerByMobile(mobile);
  if (!farmer) {
    return res.status(404).json({ error: 'No registered farmer found with this mobile number. Please register first.' });
  }

  const isPasswordValid = db.verifyPassword(password, farmer.password_hash);
  if (!isPasswordValid) {
    return res.status(401).json({ error: 'Incorrect password. Please verify and try again.' });
  }

  const token = jwt.sign(
    { farmer_id: farmer.farmer_id, mobile: farmer.mobile },
    JWT_SECRET,
    { expiresIn: '7d' }
  );

  const { password_hash, ...profile } = farmer;

  res.json({
    success: true,
    token,
    farmer: profile
  });
});

app.post('/api/auth/register', (req, res) => {
  const {
    mobile,
    password,
    name,
    district,
    village,
    land_size,
    soil_type,
    crop,
    water_source,
    water_reserve_liters,
    preferred_language
  } = req.body;

  if (!mobile || !password || !name || !district) {
    return res.status(400).json({ error: 'Name, mobile, password, and district are mandatory.' });
  }

  const existing = db.findFarmerByMobile(mobile);
  if (existing) {
    return res.status(409).json({ error: 'A farmer account with this mobile number already exists. Please login.' });
  }

  const newProfile = db.createFarmer({
    mobile,
    password,
    name,
    district,
    village: village || 'Gram Panchayat',
    land_size: Number(land_size) || 3.0,
    soil_type: soil_type || 'Black soil',
    crop: crop || 'Cotton',
    water_source: water_source || 'Borewell',
    water_reserve_liters: Number(water_reserve_liters) || 300000,
    preferred_language: preferred_language || 'kn'
  });

  const token = jwt.sign(
    { farmer_id: newProfile.farmer_id, mobile: newProfile.mobile },
    JWT_SECRET,
    { expiresIn: '7d' }
  );

  res.json({
    success: true,
    token,
    farmer: newProfile
  });
});

app.post('/api/auth/forgot-password', (req, res) => {
  const { mobile, otp, new_password } = req.body;
  if (!mobile || !otp || !new_password) {
    return res.status(400).json({ error: 'Mobile, OTP, and new password are required' });
  }

  const verifyRes = db.verifyOtp(mobile, otp);
  if (!verifyRes.valid) {
    return res.status(400).json({ error: verifyRes.message });
  }

  const updated = db.updatePassword(mobile, new_password);
  if (!updated) {
    return res.status(404).json({ error: 'Farmer account not found for this mobile' });
  }

  res.json({ success: true, message: 'Password updated successfully. You can now login.' });
});

// ----------------------------------------------------
// Authenticated Farmer Portal Routes
// ----------------------------------------------------
app.get('/api/farmer/profile', authenticateFarmer, (req: AuthenticatedRequest, res) => {
  const farmer = db.findFarmerById(req.farmerId!);
  if (!farmer) return res.status(404).json({ error: 'Farmer not found' });
  const { password_hash, ...profile } = farmer;
  res.json({ success: true, profile });
});

app.put('/api/farmer/profile', authenticateFarmer, (req: AuthenticatedRequest, res) => {
  const updates = req.body;
  delete updates.farmer_id;
  delete updates.mobile;
  delete updates.password_hash;

  const updated = db.updateFarmerProfile(req.farmerId!, updates);
  if (!updated) return res.status(404).json({ error: 'Farmer not found' });
  res.json({ success: true, profile: updated });
});

app.get('/api/farmer/dashboard', authenticateFarmer, async (req: AuthenticatedRequest, res) => {
  try {
    const farmer = db.findFarmerById(req.farmerId!);
    if (!farmer) return res.status(404).json({ error: 'Farmer not found' });

    const landSqm = calculateLandAreaSqm(farmer.land_size, farmer.land_unit);
    const weather = await fetchWeatherForDistrict(farmer.district, landSqm);

    const calculations = calculateWaterBudget(
      {
        crop: farmer.crop,
        district: farmer.district,
        land_size: farmer.land_size,
        land_unit: farmer.land_unit,
        soil_type: farmer.soil_type,
        water_source: farmer.water_source,
        water_reserve_liters: farmer.water_reserve_liters,
        growth_stage: 'Vegetative',
        season: 'Kharif',
        current_method: farmer.current_method
      },
      weather.effective_rainfall_liters
    );

    const recommendations = generateRecommendations(
      {
        crop: farmer.crop,
        district: farmer.district,
        land_size: farmer.land_size,
        land_unit: farmer.land_unit,
        soil_type: farmer.soil_type,
        water_source: farmer.water_source,
        water_reserve_liters: farmer.water_reserve_liters,
        growth_stage: 'Vegetative',
        season: 'Kharif',
        current_method: farmer.current_method
      },
      calculations
    );

    let waterAllocation = null;
    if (farmer.water_reserve_liters > 0) {
      waterAllocation = planWaterDistribution(farmer.crop, farmer.water_reserve_liters);
    }

    const viability = evaluateViability(
      {
        crop: farmer.crop,
        district: farmer.district,
        land_size: farmer.land_size,
        land_unit: farmer.land_unit,
        soil_type: farmer.soil_type,
        water_source: farmer.water_source,
        water_reserve_liters: farmer.water_reserve_liters,
        growth_stage: 'Vegetative',
        season: 'Kharif',
        current_method: farmer.current_method
      },
      calculations
    );

    const droughtMode = determineDroughtMode(
      {
        crop: farmer.crop,
        district: farmer.district,
        land_size: farmer.land_size,
        land_unit: farmer.land_unit,
        soil_type: farmer.soil_type,
        water_source: farmer.water_source,
        water_reserve_liters: farmer.water_reserve_liters,
        growth_stage: 'Vegetative',
        season: 'Kharif',
        current_method: farmer.current_method
      },
      calculations,
      weather.weekly_rainfall_total_mm
    );

    const notifications = db.getNotifications(farmer.farmer_id);
    const records = db.getIrrigationRecords(farmer.farmer_id);

    // Compute weekly usage from recent records
    const oneWeekAgo = Date.now() - 7 * 24 * 3600 * 1000;
    const weeklyUsed = records
      .filter(r => new Date(r.date).getTime() >= oneWeekAgo)
      .reduce((sum, r) => sum + r.liters_used, 0) || Math.round(calculations.daily_water_need_liters * 4.5);

    const monthlyUsed = records
      .reduce((sum, r) => sum + r.liters_used, 0) || Math.round(calculations.daily_water_need_liters * 18);

    // Next scheduled irrigation date calculation
    const nextDate = new Date();
    // If rain expected within 48h, delay by 3 days
    if (weather.irrigation_skip_days > 0) {
      nextDate.setDate(nextDate.getDate() + 3);
    } else {
      nextDate.setDate(nextDate.getDate() + 1);
    }

    const { password_hash, ...profile } = farmer;

    const dashboardData: FarmerDashboardData = {
      profile,
      weather,
      calculations,
      recommendations,
      water_allocation: waterAllocation,
      drought_mode: droughtMode,
      viability,
      notifications,
      recent_records: records.slice(0, 5),
      summary: {
        weekly_used_liters: weeklyUsed,
        monthly_used_liters: monthlyUsed,
        next_irrigation_date: nextDate.toISOString().split('T')[0],
        water_status: calculations.status.includes('Surplus')
          ? 'Surplus'
          : calculations.status.includes('Severe')
          ? 'Severe Deficit'
          : 'Moderate Deficit'
      }
    };

    res.json({ success: true, data: dashboardData });
  } catch (err) {
    console.error('Error generating farmer dashboard:', err);
    res.status(500).json({ error: 'Failed to generate farmer dashboard' });
  }
});

app.get('/api/farmer/analytics', authenticateFarmer, (req: AuthenticatedRequest, res) => {
  const analytics = db.getAnalytics(req.farmerId!);
  res.json({ success: true, data: analytics });
});

app.post('/api/farmer/irrigation-records', authenticateFarmer, (req: AuthenticatedRequest, res) => {
  const { date, crop, duration_hours, liters_used, method, notes } = req.body;
  if (!date || !liters_used) {
    return res.status(400).json({ error: 'Date and water volume in litres are required' });
  }

  const farmer = db.findFarmerById(req.farmerId!);

  const newRecord = db.addIrrigationRecord({
    farmer_id: req.farmerId!,
    date,
    crop: crop || (farmer ? farmer.crop : 'Cotton'),
    duration_hours: Number(duration_hours) || 2.5,
    liters_used: Number(liters_used),
    method: method || (farmer ? farmer.current_method : 'Drip Irrigation'),
    notes: notes || ''
  });

  const updatedFarmer = db.findFarmerById(req.farmerId!);

  res.json({
    success: true,
    record: newRecord,
    updated_reserve_liters: updatedFarmer?.water_reserve_liters
  });
});

app.delete('/api/farmer/irrigation-records/:id', authenticateFarmer, (req: AuthenticatedRequest, res) => {
  const recordId = req.params.id;
  const deleted = db.deleteIrrigationRecord(req.farmerId!, recordId);
  if (!deleted) return res.status(404).json({ error: 'Record not found' });
  res.json({ success: true, message: 'Record deleted' });
});

// Context-Aware Multilingual AI Farm Assistant
app.post('/api/farmer/ai-chat', authenticateFarmer, async (req: AuthenticatedRequest, res) => {
  const { message, language } = req.body;
  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'Message is required' });
  }

  const farmer = db.findFarmerById(req.farmerId!);
  if (!farmer) return res.status(404).json({ error: 'Farmer not found' });

  const analytics = db.getAnalytics(req.farmerId!);
  const records = db.getIrrigationRecords(req.farmerId!);
  const lang = (language || farmer.preferred_language || 'kn') as SupportedLanguage;

  const totalUsedWeek = analytics.water_usage_comparison[0]?.actual_used || 18000;
  const weeklyTarget = analytics.water_usage_comparison[0]?.recommended || 24000;
  const reserve = farmer.water_reserve_liters;

  // Language display name
  const langNames: Record<SupportedLanguage, string> = {
    en: 'English',
    kn: 'Kannada (ಕನ್ನಡ)',
    hi: 'Hindi (हिन्दी)',
    te: 'Telugu (తెలుగు)',
    ta: 'Tamil (தமிழ்)',
    ml: 'Malayalam (മലയാളം)'
  };

  const targetLang = langNames[lang] || 'Kannada';

  // Check if Gemini can be called (not in cooldown and client configured)
  if (!isGeminiInCooldown() && aiClient && process.env.GEMINI_API_KEY) {
    try {
      const prompt = `You are the WaterWise Personal Agricultural AI Assistant for a farmer in Karnataka, India.
Here is the real verified farm profile and data for the logged-in farmer:
- Farmer Name: ${farmer.name}
- Farmer ID: ${farmer.farmer_id}
- District: ${farmer.district}
- Village: ${farmer.village}
- Farm Size: ${farmer.land_size} ${farmer.land_unit}
- Soil: ${farmer.soil_type}
- Primary Crop: ${farmer.crop}
- Current Irrigation Method: ${farmer.current_method}
- Available Water Reserve: ${reserve.toLocaleString('en-IN')} Litres
- Recorded Water Used This Week: ${totalUsedWeek.toLocaleString('en-IN')} Litres (Budget: ${weeklyTarget.toLocaleString('en-IN')} L)
- Total Irrigation Events Logged: ${records.length} records

Farmer's Question: "${message}"

INSTRUCTIONS:
1. Respond exclusively in ${targetLang}. Use simple, respectful, farmer-friendly terms.
2. Ground your answer in their actual farm records provided above.
3. If they ask about water usage, cite their real numbers (${totalUsedWeek.toLocaleString('en-IN')} L used, ${reserve.toLocaleString('en-IN')} L in reserve).
4. Provide practical, high-value agronomic advice (e.g. night watering, mulching, checking borewell pressure, rain postponement).
5. Keep your response under 80 words. Be concise, clear, and reassuring.`;

      const genPromise = aiClient.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
      });

      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('AI Assistant Timeout')), 3500)
      );

      const aiResponse = await Promise.race([genPromise, timeoutPromise]);
      const replyText = aiResponse.text?.trim();

      if (replyText) {
        return res.json({
          success: true,
          reply: replyText,
          grounded_data: `${farmer.crop} in ${farmer.district} (${farmer.land_size} ac, Reserve: ${reserve.toLocaleString('en-IN')}L)`
        });
      }
    } catch (err) {
      handleGeminiError(err, 'farmer ai-chat');
    }
  }

  // Multilingual agronomic fallback engine grounded in farmer's live data
  let fallbackReply = '';
  const lowerMsg = message.toLowerCase();

  const isUsageQ = lowerMsg.includes('week') || lowerMsg.includes('water') || lowerMsg.includes('usage') ||
    lowerMsg.includes('used') || lowerMsg.includes('ನೀರು') || lowerMsg.includes('ಬಳಕೆ') || lowerMsg.includes('ವಾರ') ||
    lowerMsg.includes('पानी') || lowerMsg.includes('उपयोग') || lowerMsg.includes('ఖర్చు') || lowerMsg.includes('பயன்பாடு');

  const isTimingQ = lowerMsg.includes('when') || lowerMsg.includes('time') || lowerMsg.includes('schedule') ||
    lowerMsg.includes('night') || lowerMsg.includes('morning') || lowerMsg.includes('ಯಾವಾಗ') || lowerMsg.includes('ಸಮಯ') ||
    lowerMsg.includes('ವೇಳೆ') || lowerMsg.includes('कब') || lowerMsg.includes('समय') || lowerMsg.includes('ఎప్పుడు');

  const isSavingQ = lowerMsg.includes('save') || lowerMsg.includes('mulch') || lowerMsg.includes('drought') ||
    lowerMsg.includes('ಉಳಿಸು') || lowerMsg.includes('ಹೊದಿಕೆ') || lowerMsg.includes('ಬರ') || lowerMsg.includes('ಬಚತ್') ||
    lowerMsg.includes('बचाव') || lowerMsg.includes('सूखा') || lowerMsg.includes('సేవ్') || lowerMsg.includes('சேமிக்க');

  const isFertigationQ = lowerMsg.includes('fertilizer') || lowerMsg.includes('fertigation') || lowerMsg.includes('urea') ||
    lowerMsg.includes('dap') || lowerMsg.includes('ಗೊಬ್ಬರ') || lowerMsg.includes('ಪೋಷಕಾಂಶ') || lowerMsg.includes('खाद') ||
    lowerMsg.includes('ಎರುವು');

  const isBorewellQ = lowerMsg.includes('borewell') || lowerMsg.includes('motor') || lowerMsg.includes('pump') ||
    lowerMsg.includes('ಕೊಳವೆಬಾವಿ') || lowerMsg.includes('ಮೋಟಾರ್') || lowerMsg.includes('बोरवेल') || lowerMsg.includes('మోటారు');

  if (isUsageQ) {
    if (lang === 'kn') {
      fallbackReply = `ನಮಸ್ಕಾರ ${farmer.name} ಅವರೇ, ನಿಮ್ಮ ಕೃಷಿ ದಾಖಲೆಗಳ ಪ್ರಕಾರ ಈ ವಾರ ನೀವು ಸುಮಾರು ${totalUsedWeek.toLocaleString('en-IN')} ಲೀಟರ್ ನೀರನ್ನು ಬಳಸಿದ್ದೀರಿ (ನಿಗದಿತ ಬಜೆಟ್: ${weeklyTarget.toLocaleString('en-IN')} ಲೀಟರ್). ನಿಮ್ಮ ಸಂಗ್ರಹದಲ್ಲಿ ಇನ್ನೂ ${reserve.toLocaleString('en-IN')} ಲೀಟರ್ ಲಭ್ಯವಿದೆ. ${farmer.crop} ಬೆಳೆಗೆ ಹನಿ ನೀರಾವರಿಯನ್ನು ನಿಗದಿತ ಅಂತರದಲ್ಲಿ ಮುಂದುವರಿಸಿ.`;
    } else if (lang === 'hi') {
      fallbackReply = `नमस्ते ${farmer.name} जी, आपके फार्म रिकॉर्ड के अनुसार इस सप्ताह आपने लगभग ${totalUsedWeek.toLocaleString('en-IN')} लीटर पानी का उपयोग किया है (लक्ष्य: ${weeklyTarget.toLocaleString('en-IN')} लीटर)। आपके पास अभी ${reserve.toLocaleString('en-IN')} लीटर जल भंडार उपलब्ध है। ${farmer.crop} के लिए ड्रिप सिंचाई जारी रखें।`;
    } else if (lang === 'te') {
      fallbackReply = `నమస్కారం ${farmer.name} గారు, మీ రికార్డుల ప్రకారం ఈ వారం మీరు ${totalUsedWeek.toLocaleString('en-IN')} లీటర్ల నీటిని ఉపయోగించారు. మీ నిల్వలో ఇంకా ${reserve.toLocaleString('en-IN')} లీటర్ల నీరు అందుబాటులో ఉంది.`;
    } else if (lang === 'ta') {
      fallbackReply = `வணக்கம் ${farmer.name}, உங்கள் பண்ணை பதிவுகளின்படி இந்த வாரம் நீங்கள் சுமார் ${totalUsedWeek.toLocaleString('en-IN')} லிட்டர் தண்ணீரைப் பயன்படுத்தியுள்ளீர்கள். உங்கள் இருப்பில் ${reserve.toLocaleString('en-IN')} லிட்டர் உள்ளது.`;
    } else if (lang === 'ml') {
      fallbackReply = `നമസ്കാരം ${farmer.name}, നിങ്ങളുടെ ഫാം രേഖകൾ പ്രകാരം ഈ ആഴ്ച നിങ്ങൾ ${totalUsedWeek.toLocaleString('en-IN')} ലിറ്റർ വെള്ളം ഉപയോഗിച്ചു. നിങ്ങളുടെ റിസർവിൽ ${reserve.toLocaleString('en-IN')} ലിറ്റർ ലഭ്യമാണ്.`;
    } else {
      fallbackReply = `Hello ${farmer.name}, based on your farm records, you have consumed approximately ${totalUsedWeek.toLocaleString('en-IN')} Litres this week (budget: ${weeklyTarget.toLocaleString('en-IN')} L). Your current available water reserve is ${reserve.toLocaleString('en-IN')} Litres for your ${farmer.crop}.`;
    }
  } else if (isTimingQ) {
    if (lang === 'kn') {
      fallbackReply = `${farmer.name} ಅವರೇ, ${farmer.district} ಹವಾಮಾನದಲ್ಲಿ ಸಂಜೆ 6:30 ರ ನಂತರ ಅಥವಾ ಬೆಳಗಿನ ಜಾವ ನೀರು ಹಾಯಿಸುವುದು ಅತ್ಯುತ್ತಮ. ಇದರಿಂದ ಬಿಸಿಲಿನ ಆವಿಯಾಗುವಿಕೆ ಶೇ.25 ರಷ್ಟು ಕಡಿಮೆಯಾಗುತ್ತದೆ ಮತ್ತು ಬೇರುಗಳು ತೇವಾಂಶವನ್ನು ಪೂರ್ಣವಾಗಿ ಹೀರಿಕೊಳ್ಳುತ್ತವೆ.`;
    } else if (lang === 'hi') {
      fallbackReply = `${farmer.name} जी, ${farmer.district} में शाम 6:30 के बाद या भोर में सिंचाई करना सबसे अच्छा है ताकि धूप से होने वाले 25% वाष्पीकरण को रोका जा सके।`;
    } else {
      fallbackReply = `For your ${farmer.crop} in ${farmer.district}, irrigate in the evening after 6:30 PM or early morning to eliminate up to 25% evaporative loss under hot semi-arid conditions.`;
    }
  } else if (isSavingQ) {
    if (lang === 'kn') {
      fallbackReply = `ನೀರು ಉಳಿಸಲು 3 ಪ್ರಮುಖ ಹಂತಗಳು: 1) ಸಾಲುಗಳ ನಡುವೆ ಒಣ ಹುಲ್ಲಿನ ಹೊದಿಕೆ (ಮಲ್ಚಿಂಗ್) ಹಾಕಿ, 2) ಹನಿ ನೀರಾವರಿ (M01) ಅಳವಡಿಸಿ ಶೇ.40 ನೀರು ಉಳಿಸಿ, 3) ತೀವ್ರ ಕೊರತೆ ಇದ್ದರೆ ಪರ್ಯಾಯ ಸಾಲು ನೀರಾವರಿ (M18) ಅನುಸರಿಸಿ.`;
    } else if (lang === 'hi') {
      fallbackReply = `पानी बचाने के लिए: 1) क्यारियों पर सूखी घास की मल्चिंग करें, 2) ड्रिप सिंचाई अपनाकर 40% पानी बचाएं, 3) पानी की कमी होने पर एक छोड़कर एक नाली में पानी दें (M18)।`;
    } else {
      fallbackReply = `Top 3 water-saving measures for ${farmer.crop}: 1) Apply 5-8 cm organic mulch (M03) to halt soil evaporation, 2) Use drip irrigation (M01) for 45% savings, 3) Adopt alternate furrow irrigation (M18) during dry spells.`;
    }
  } else if (isFertigationQ) {
    if (lang === 'kn') {
      fallbackReply = `${farmer.name} ಅವರೇ, ಗೊಬ್ಬರವನ್ನು ನೇರವಾಗಿ ಹನಿ ನೀರಾವರಿ ವೆಂಚುರಿ ಮೂಲಕ (ಫರ್ಟಿಗೇಷನ್) ನೀಡುವುದರಿಂದ ಪೋಷಕಾಂಶಗಳು ನೇರವಾಗಿ ಬೇರಿಗೆ ತಲುಪುತ್ತವೆ ಮತ್ತು ರಸಗೊಬ್ಬರ ವ್ಯರ್ಥವಾಗುವುದು ಶೇ.30 ರಷ್ಟು ಕಡಿಮೆಯಾಗುತ್ತದೆ.`;
    } else {
      fallbackReply = `Deliver water-soluble fertilizers directly through your drip system (fertigation) in the middle 60% of the irrigation run to achieve 90% nutrient uptake and minimize deep leaching.`;
    }
  } else if (isBorewellQ) {
    if (lang === 'kn') {
      fallbackReply = `ಕೊಳವೆಬಾವಿಯ ನೀರಿನ ಮಟ್ಟ ಕಾಪಾಡಲು ಮೋಟಾರನ್ನು ನಿರಂತರವಾಗಿ ಓಡಿಸದೆ 2-3 ಗಂಟೆಗಳ ವಿರಾಮ ನೀಡಿ. ರಾತ್ರಿ ವೇಳೆ ಸರಬರಾಜಾಗುವ ಕೃಷಿ ವಿದ್ಯುತ್ ಅವಧಿಯಲ್ಲಿ ಪಂಪ್ ಚಲಾಯಿಸಿ.`;
    } else {
      fallbackReply = `To safeguard borewell yield in ${farmer.district}, split pumping into 2-3 hour cycles rather than continuous draw, preventing casing drawdown and aquifer stress.`;
    }
  } else {
    if (lang === 'kn') {
      fallbackReply = `ನಮಸ್ಕಾರ ${farmer.name} ಅವರೇ! ನಿಮ್ಮ ${farmer.district} ಜಮೀನಿನ ${farmer.crop} ಬೆಳೆ, ಮಣ್ಣು (${farmer.soil_type}) ಮತ್ತು ನೀರಿನ ಸಂಗ್ರಹದ (${reserve.toLocaleString('en-IN')}L) ಆಧಾರದ ಮೇಲೆ ನಾನು ಸಲಹೆ ನೀಡಬಲ್ಲೆ. ನೀವು ನೀರಾವರಿ ವೇಳಾಪಟ್ಟಿ, ಗೊಬ್ಬರ ಅಥವಾ ನೀರು ಉಳಿತಾಯದ ಬಗ್ಗೆ ಕೇಳಬಹುದು.`;
    } else if (lang === 'hi') {
      fallbackReply = `नमस्ते ${farmer.name} जी! मैं आपके ${farmer.district} खेत की ${farmer.crop} फसल और ${reserve.toLocaleString('en-IN')}L जल भंडार के आधार पर सलाह दे सकता हूँ। आप कोई भी सवाल पूछ सकते हैं।`;
    } else {
      fallbackReply = `Hello ${farmer.name}! I am monitoring your ${farmer.crop} farm in ${farmer.district} (${(reserve / 1000).toFixed(0)}k L reserve). Feel free to ask about your irrigation schedule, water budget, mulching, or weather advisories.`;
    }
  }

  res.json({
    success: true,
    reply: fallbackReply,
    grounded_data: `${farmer.crop} in ${farmer.district} (${farmer.land_size} ac, Reserve: ${reserve.toLocaleString('en-IN')}L)`
  });
});

// Setup Vite or static serving
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`WaterWise Server is listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();

