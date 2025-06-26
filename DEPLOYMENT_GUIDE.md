# COE Platform Deployment & Sleep Prevention Guide

## Current Status
✅ **Keep-alive service is ACTIVE** - Automatically pings every 15 minutes
✅ **Health check endpoint available** - `/health_check` for external monitoring
✅ **Optimized Streamlit configuration** - Enhanced performance settings

## Recommended Solutions (in order of effectiveness)

### 1. UptimeRobot Monitoring (Recommended - Free)
**Setup Steps:**
1. Go to [uptimerobot.com](https://uptimerobot.com) and create free account
2. Add "New Monitor" with these settings:
   - Monitor Type: HTTP(s)
   - URL: Your Streamlit app URL
   - Monitoring Interval: 10 minutes
   - Timeout: 30 seconds
3. Set up email alerts for downtime notifications

**Benefits:**
- External monitoring prevents sleeping
- Free tier allows 50 monitors
- Sends alerts if app goes down
- Works with any hosting platform

### 2. GitHub Actions Keep-Alive (Advanced)
Create `.github/workflows/keep-alive.yml`:
```yaml
name: Keep COE Platform Alive
on:
  schedule:
    - cron: '*/10 * * * *'  # Every 10 minutes
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping App
        run: |
          curl -f "YOUR_APP_URL" || echo "App might be sleeping"
          curl -f "YOUR_APP_URL/health_check" || echo "Health check failed"
```

### 3. Streamlit Cloud Optimization
**Current Configuration:**
- Keep-alive service running internally
- Optimized server settings for responsiveness
- Session state management for faster loading
- Health check endpoint for monitoring

### 4. Alternative Hosting Options

**A) Heroku (Paid - Most Reliable)**
- Cost: $7/month for Hobby tier
- Benefits: No sleeping, custom domains, better performance
- Setup: `git push heroku main` deployment

**B) Railway (Affordable)**
- Cost: $5/month for Hobby plan
- Benefits: No sleeping, generous resource limits
- Setup: Connect GitHub repository

**C) DigitalOcean App Platform**
- Cost: $5/month for basic tier
- Benefits: Dedicated resources, no sleeping
- Setup: Docker or GitHub deployment

## Current Implementation Details

### Keep-Alive Service
```python
# Automatically started in app.py
# Pings every 15 minutes: http://0.0.0.0:5000
# Logs success/failure in console
```

### Health Check Endpoint
```
URL: /health_check
Purpose: Lightweight monitoring endpoint
Response: JSON health status
Auto-refresh: Optional 5-minute intervals
```

### Monitoring Setup
1. **Internal**: Keep-alive service running automatically
2. **External**: Set up UptimeRobot for additional monitoring
3. **Alerts**: Configure notifications for downtime
4. **Logs**: Monitor console for ping success/failure

## Quick Start (5 minutes)
1. ✅ Keep-alive already running in your app
2. 📋 Sign up for UptimeRobot (free)
3. 🔗 Add your app URL to monitoring
4. 📧 Set up email notifications
5. ✅ Done - your app will stay awake

## Troubleshooting

**If app still sleeps:**
- Check UptimeRobot is properly configured
- Verify keep-alive logs show successful pings
- Consider upgrading to paid hosting
- Contact Streamlit support for persistent issues

**Monitoring checklist:**
- [ ] Keep-alive service shows successful pings
- [ ] UptimeRobot monitor is active
- [ ] Health check endpoint responds
- [ ] App loads quickly after inactivity

The combination of internal keep-alive + external monitoring should eliminate sleeping issues completely.