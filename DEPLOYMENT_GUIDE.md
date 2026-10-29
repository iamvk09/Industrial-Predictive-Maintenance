# 🚀 Deployment Guide – Hugging Face Spaces

**Project:** Industrial Predictive Maintenance (LightGBM + SHAP)  
**Estimated deployment time:** 15–25 minutes  
**Cost:** 100% FREE

---

## ✅ Pre-Deployment Checklist

Make sure you have these files ready:

- [ ] `app.py`
- [ ] `requirements.txt`
- [ ] `lgbm_model.pkl`
- [ ] `README.md`
- [ ] `.gitignore`

---

## 📋 Step-by-Step Deployment

### STEP 1 – Create a Hugging Face Account

1. Go to [https://huggingface.co](https://huggingface.co)
2. Click **Sign Up** (top right)
3. Fill in your username, email, and password
4. Verify your email address
5. You're in! ✅

> **Tip:** Choose a professional username — it appears in your Space's public URL.

---

### STEP 2 – Create a New Space

1. Log in to Hugging Face
2. Click your **profile picture** (top right) → **New Space**

   OR go directly to: [https://huggingface.co/new-space](https://huggingface.co/new-space)

3. Fill in the form:

   | Field | Value |
   |-------|-------|
   | **Space name** | `predictive-maintenance-lgbm` (or any name you like) |
   | **License** | MIT |
   | **SDK** | **Gradio** ← select this |
   | **SDK Version** | 4.44.0 (or latest) |
   | **Visibility** | Public (for portfolio sharing) |

4. Click **Create Space** ✅

---

### STEP 3 – Upload Your Files

You have **two options** – choose whichever feels easier:

#### Option A: Upload via Web Interface (Easiest)

1. Inside your new Space, click the **Files** tab
2. Click **Add file → Upload files**
3. Upload these files **one at a time** or drag them all at once:
   - `app.py`
   - `requirements.txt`
   - `lgbm_model.pkl`
   - `README.md`
4. After each upload, click **Commit changes to main**

#### Option B: Upload via Git (Faster for multiple files)

```bash
# 1. Install Git LFS (needed for the .pkl model file)
git lfs install

# 2. Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/predictive-maintenance-lgbm
cd predictive-maintenance-lgbm

# 3. Copy your files into the cloned folder
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/lgbm_model.pkl .
cp /path/to/README.md .
cp /path/to/.gitignore .

# 4. Track the model file with Git LFS
git lfs track "*.pkl"
git add .gitattributes

# 5. Stage, commit, and push
git add .
git commit -m "Initial deployment: LightGBM predictive maintenance app"
git push

# You'll be prompted for your HF username and access token (see below)
```

**To get your HF access token:**
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → name it anything → Role: **Write**
3. Copy the token — use it as your password when Git prompts

---

### STEP 4 – Verify the README Header

The `README.md` must start with this YAML block for Hugging Face to configure your Space correctly. Open your uploaded README and confirm the first lines are:

```yaml
---
title: Industrial Predictive Maintenance - LightGBM + SHAP
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
```

If it's missing, click the file in the Files tab, click the **pencil (Edit)** icon, paste the block at the top, and commit.

---

### STEP 5 – Watch the Build

1. Click the **App** tab in your Space
2. You'll see a **Building** spinner — this takes 3–8 minutes the first time
3. Watch the build logs for any errors (click **Logs** if visible)

**Build complete?** The spinner disappears and your app loads. 🎉

---

### STEP 6 – Test Your Deployment

Once the app is live, run these checks:

#### ✅ Single Prediction Test
1. Open the **🔍 Single Prediction** tab
2. Click **⚠️ High-Risk Scenario** button
3. Click **🔮 Predict**
4. Expected: Red "FAILURE PREDICTED" box, >70% probability, SHAP chart shown

#### ✅ Normal Operation Test
1. Click **✅ Normal Operation** button
2. Click **🔮 Predict**
3. Expected: Green "NORMAL OPERATION" box, <40% probability

#### ✅ Batch Prediction Test
1. Create a test CSV with 5 rows using these columns:
   ```
   Type,Air_temp,Process_temp,Rotation_speed,Torque,Tool_wear
   1,300,310,1500,40,100
   0,302,312,1200,75,220
   2,298,308,2000,25,50
   1,301,311,1400,65,180
   0,303,313,1300,70,200
   ```
2. Upload in the **📂 Batch Prediction** tab
3. Click **⚡ Run Batch Prediction**
4. Verify summary stats appear and results download

---

### STEP 7 – Share Your Space

Your Space URL follows this pattern:

```
https://huggingface.co/spaces/YOUR_USERNAME/predictive-maintenance-lgbm
```

**Add it to:**
- 📌 LinkedIn profile (Projects section)
- 📌 GitHub README
- 📌 Your university project report
- 📌 Resume / CV

---

## 🔧 Troubleshooting

### ❌ "Module not found" during build

**Problem:** A package in `requirements.txt` failed to install  
**Fix:** Check the build logs. If a package name is wrong, edit `requirements.txt` and commit again.

---

### ❌ "lgbm_model.pkl not found" / App runs in demo mode

**Problem:** The model file wasn't uploaded, or Git LFS wasn't used  
**Fix:**
1. Check the **Files** tab — is `lgbm_model.pkl` listed?
2. If it shows as 134 bytes (pointer file), Git LFS isn't tracking it:
   ```bash
   git lfs track "*.pkl"
   git add .gitattributes
   git add lgbm_model.pkl
   git commit -m "Fix: add model via LFS"
   git push
   ```
3. If not listed, re-upload via the web interface

---

### ❌ "Error reading CSV" in batch prediction

**Problem:** Column names don't match exactly  
**Fix:** Column matching is case-insensitive and flexible. Ensure your CSV has at least these keywords in column names: `type`, `air`, `process`, `rotational`, `torque`, `tool`

---

### ❌ Build timeout / space stuck on "Building"

**Fix:**
1. Click **Settings** → **Factory Reboot Space**
2. If it still fails, check the **Logs** tab for the specific error

---

### ❌ App crashes immediately after loading

**Fix:**
1. Verify `app_file: app.py` in your README.md YAML header
2. Check that all packages in `requirements.txt` are compatible
3. Look in **Logs** for the Python traceback

---

## 🔄 Updating Your App

To update your deployed app after making changes:

**Via web:** Edit any file directly in the Files tab → Commit → Space rebuilds automatically

**Via Git:**
```bash
git add app.py           # or whichever file changed
git commit -m "Update: improved SHAP visualisation"
git push
# Space rebuilds in ~3 minutes
```

---

## 📊 Space Settings (Optional Tweaks)

In **Settings** tab of your Space:

- **Sleep setting:** "Never sleep" keeps it always available (but uses more quota)
- **Custom domain:** You can add a custom URL (paid feature)
- **Secrets:** For API keys — not needed for this project

---

## 🎓 For Your Presentation

Before presenting, open the app and:
1. Run the **High-Risk** example → show the failure prediction + SHAP chart
2. Open **Model Information** → show the performance comparison chart
3. Run a batch upload → show the summary dashboard
4. Show the **About** tab for team/university info

The live URL works on any device — you can demonstrate it on a projector, phone, or laptop without installing anything.

---

## 📞 Need Help?

- Hugging Face Docs: [https://huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- Gradio Docs: [https://www.gradio.app/docs](https://www.gradio.app/docs)
- Hugging Face Discord: [https://discord.gg/huggingface](https://discord.gg/huggingface)

---

*Deployment guide for RVITM Predictive Maintenance Project – Vibhor Kumar, Rahul Shankar Panday, Ridesh Rajesh*
