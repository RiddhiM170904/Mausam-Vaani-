# PWA Icons Setup

To complete the PWA installation, you need to add the following icon files to this directory:

## Required Icon Sizes:
- `icon-16x16.png` (16x16 pixels)
- `icon-32x32.png` (32x32 pixels)
- `icon-72x72.png` (72x72 pixels)
- `icon-96x96.png` (96x96 pixels)
- `icon-128x128.png` (128x128 pixels)
- `icon-144x144.png` (144x144 pixels)
- `icon-152x152.png` (152x152 pixels)
- `icon-192x192.png` (192x192 pixels)
- `icon-384x384.png` (384x384 pixels)
- `icon-512x512.png` (512x512 pixels)

## Icon Design Guidelines:
- Use the Mausam Vaani logo or weather-related iconography
- Ensure icons are square (1:1 aspect ratio)
- Use high contrast colors that work on both light and dark backgrounds
- Follow the app's color scheme (blues and purples)
- Make sure icons are recognizable at small sizes

## Tools to Generate Icons:
1. **Canva** - Easy icon creation with templates
2. **Figma** - Professional design tool
3. **PWA Builder** - Microsoft's tool for PWA assets
4. **Real Favicon Generator** - Automatic icon generation
5. **App Icon Generator** - Batch icon generation

## Quick Setup:
1. Create a 512x512 master icon
2. Use an online icon generator to create all sizes
3. Place all generated icons in this `/public/icons/` directory
4. Ensure filenames match exactly what's referenced in manifest.json

The app will work without these icons, but they're required for a complete PWA experience and proper app installation.