# 🚀 Deploying Mausam-Vaani AI-Backend to Vercel

## ⚠️ Important Limitations

Vercel has strict limitations for Python ML apps:
- **Execution Timeout**: 10s (Hobby), 60s (Pro)
- **Cold Start**: 5-10s (affects first request)
- **Package Size**: 50MB limit (PyTorch is ~200MB)
- **Memory**: Limited (3GB max on Pro)

**Recommended**: Use **Railway**, **Render**, or **Google Cloud Run** for ML workloads instead.

---

## 📋 Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally
3. **API Keys**: 
   - Google Gemini API key
   - OpenWeather API key (optional, uses synthetic data if missing)

---

## 🛠️ Step-by-Step Deployment

### 1️⃣ Install Vercel CLI

```powershell
npm install -g vercel
```

### 2️⃣ Login to Vercel

```powershell
vercel login
```

### 3️⃣ Navigate to AI-Backend Directory

```powershell
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\AI-Backend"
```

### 4️⃣ Deploy to Vercel

```powershell
vercel
```

Follow the prompts:
- **Set up and deploy?** → Yes
- **Which scope?** → Your account
- **Link to existing project?** → No
- **Project name?** → mausam-vaani-ai (or your choice)
- **Directory?** → ./ (current directory)
- **Override settings?** → No

### 5️⃣ Add Environment Variables

**Option A: Via CLI**
```powershell
vercel env add GEMINI_API_KEY
# Paste your Gemini API key when prompted
# Select: Production, Preview, Development (all)

vercel env add OPENWEATHER_API_KEY
# Paste your OpenWeather API key when prompted
# Select: Production, Preview, Development (all)
```

**Option B: Via Dashboard**
1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Select your project → Settings → Environment Variables
3. Add:
   - `GEMINI_API_KEY` = your_gemini_key
   - `OPENWEATHER_API_KEY` = your_openweather_key
4. Save and redeploy

### 6️⃣ Deploy Production Build

```powershell
vercel --prod
```

---

## 🌐 Access Your API

After deployment, you'll get a URL like:
```
https://mausam-vaani-ai.vercel.app
```

**Test endpoints:**
- Health check: `https://your-url.vercel.app/`
- Weather prediction: `https://your-url.vercel.app/predict`
- Model diagnostics: `https://your-url.vercel.app/model-diagnostics`

---

## 🔧 Update Frontend API URL

Update `Frontend/src/config/weatherApi.js`:

```javascript
const API_BASE_URL = 'https://your-vercel-url.vercel.app';
```

---

## 🐛 Troubleshooting

### Error: "Function size exceeds limit"
**Solution**: PyTorch is too large. Options:
1. Use **torch-cpu** (smaller): Add to requirements.txt
   ```
   --extra-index-url https://download.pytorch.org/whl/cpu
   torch==2.5.1+cpu
   ```
2. Use **Railway** or **Render** instead (recommended)

### Error: "Execution timeout"
**Solution**: 
- Upgrade to Vercel Pro (60s timeout)
- Or use Railway/Render (no timeout limits)

### Error: "Module not found"
**Solution**: Check `requirements.txt` has all dependencies
```powershell
vercel logs
```

### Cold starts taking too long
**Solution**: 
- Keep dummy predictions lightweight
- Consider serverless alternatives (Railway keeps containers warm)

---

## 🎯 Alternative Deployment (Recommended)

### Railway (Recommended for ML)
```powershell
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway init
railway up
```

**Advantages**:
- No timeout limits
- Larger package sizes
- Persistent containers (no cold starts)
- Free tier available

### Render
1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repo → AI-Backend folder
4. Build: `pip install -r requirements.txt`
5. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

## 📊 Vercel Deployment Checklist

- [ ] Vercel CLI installed
- [ ] Logged into Vercel account
- [ ] `vercel.json` created
- [ ] `requirements.txt` updated
- [ ] `api/index.py` entry point created
- [ ] `.vercelignore` configured
- [ ] Environment variables added (GEMINI_API_KEY, OPENWEATHER_API_KEY)
- [ ] Deployed with `vercel --prod`
- [ ] Frontend API URL updated
- [ ] Tested API endpoints

---

## 💡 Tips

1. **Monitor Usage**: Check Vercel dashboard for function execution times
2. **Logs**: Use `vercel logs` to debug issues
3. **Local Testing**: Run `vercel dev` to test serverless locally
4. **Cost**: Vercel Hobby is free but limited; upgrade to Pro if needed
5. **Consider Railway**: Better suited for ML/AI workloads

---

## 🚨 Known Issues

- ❌ PyTorch may exceed 50MB limit → Use torch-cpu or Railway
- ❌ Cold starts slow (~10s) → Use Railway (keeps warm)
- ❌ Model file too large → Using dummy predictions only
- ⚠️ First request slow → Expected on serverless

---

## 📞 Need Help?

- Vercel Docs: https://vercel.com/docs
- Railway Docs: https://docs.railway.app
- GitHub Issues: Report problems in repo

**Recommendation**: For production ML deployment, use **Railway** or **Google Cloud Run** instead of Vercel.
