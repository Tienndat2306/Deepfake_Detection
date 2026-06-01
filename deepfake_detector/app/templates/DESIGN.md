---
name: Synthetic Intelligence Guardian
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#3a3939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353534'
  on-surface: '#e5e2e1'
  on-surface-variant: '#bbc9cf'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#859399'
  outline-variant: '#3c494e'
  surface-tint: '#4cd6ff'
  primary: '#a4e6ff'
  on-primary: '#003543'
  primary-container: '#00d1ff'
  on-primary-container: '#00566a'
  inverse-primary: '#00677f'
  secondary: '#d7ffc5'
  on-secondary: '#053900'
  secondary-container: '#2ff801'
  on-secondary-container: '#0f6d00'
  tertiary: '#ffd2cc'
  on-tertiary: '#690003'
  tertiary-container: '#ffaba0'
  on-tertiary-container: '#a20007'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#b7eaff'
  primary-fixed-dim: '#4cd6ff'
  on-primary-fixed: '#001f28'
  on-primary-fixed-variant: '#004e60'
  secondary-fixed: '#79ff5b'
  secondary-fixed-dim: '#2ae500'
  on-secondary-fixed: '#022100'
  on-secondary-fixed-variant: '#095300'
  tertiary-fixed: '#ffdad5'
  tertiary-fixed-dim: '#ffb4aa'
  on-tertiary-fixed: '#410001'
  on-tertiary-fixed-variant: '#930005'
  background: '#131313'
  on-background: '#e5e2e1'
  surface-variant: '#353534'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '500'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.05em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.03em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 24px
  margin-desktop: 40px
  margin-mobile: 16px
  container-max: 1440px
---

## Brand & Style
This design system centers on forensic precision and high-tech security. The brand personality is authoritative, vigilant, and technically advanced, targeting cybersecurity analysts, media forensic experts, and high-stakes verification teams. 

The visual direction is **Cyber-Modern with Glassmorphism**. It utilizes deep charcoal foundations to minimize eye strain during intense video analysis, while employing high-contrast "cyber" accents to draw immediate attention to critical detection data. The aesthetic balances the transparency of glass elements—representing the "unmasking" of deepfakes—with the rigid structure of a professional forensic tool.

## Colors
The palette is engineered for a dark-room forensic environment. 

*   **Primary (Cyber Blue):** Used for interactive elements, progress indicators, and "Neutral" or "Processing" states. It evokes a sense of high-tech scanning.
*   **Secondary (Neon Green):** Exclusively reserved for "Authentic" or "Safe" status results.
*   **Tertiary (Alert Red):** Used for "Deepfake Detected" alerts and critical system warnings.
*   **Neutral (Deep Charcoal):** A layered set of blacks and dark greys (#0A0A0A to #1A1A1A) provides the structural base.

Surface colors should use semi-transparent hex values (e.g., `#FFFFFF0D`) when applied to glassmorphic containers to allow background blurs to bleed through.

## Typography
The system uses **Inter** for all primary interface copy to ensure maximum legibility and a modern, neutral tone. For data-heavy readouts, frame timestamps, and confidence scores, **JetBrains Mono** is utilized to provide a "technical/coded" feel that distinguishes raw data from UI labels.

Headlines should remain tight and bold. Labels and technical data points should use uppercase styling with slight letter spacing to enhance the forensic aesthetic.

## Layout & Spacing
The design system follows a **12-column fluid grid** for desktop, collapsing to a **4-column grid** for mobile. 

A strict 4px base unit governs all padding and margins. In the analysis dashboard, high-density layouts are preferred to allow experts to view the video player, frame grid, and metadata simultaneously. 

*   **Desktop:** 40px outer margins, 24px gutters.
*   **Mobile:** 16px outer margins, 16px gutters.
*   **Components:** Use "Compact" spacing (8px-12px) for data tables and "Spacious" spacing (24px-32px) for landing sections and upload zones.

## Elevation & Depth
Depth is achieved through **Glassmorphism** and tonal layering rather than traditional shadows.

1.  **Background:** Pure `#0A0A0A`.
2.  **Base Layer:** Solid `#121212` with a 1px border of `#FFFFFF1A`.
3.  **Glass Panels:** Semi-transparent surfaces (`#FFFFFF0A`) with a `20px` backdrop-blur. 
4.  **Interactive Glows:** Instead of drop shadows, use subtle outer glows (`0px 0px 15px`) in Cyber Blue (#00D1FF) for active states or highlighted detections.

All panels must feature a 1px "inner-glow" or "top-light" border to simulate light hitting the edge of a glass pane.

## Shapes
The shape language is **Soft (0.25rem - 0.75rem)**. 

While the aesthetic is "high-tech," completely sharp corners feel dated; slight rounding suggests a refined, modern software experience. 
*   Standard buttons and inputs: `4px` (rounded-sm).
*   Glass cards and video containers: `12px` (rounded-lg).
*   Status chips: `100px` (full pill) to differentiate them from functional buttons.

## Components
*   **Glassmorphism Cards:** Must use a background of `rgba(255, 255, 255, 0.05)`, a backdrop-filter of `blur(20px)`, and a 1px border of `rgba(255, 255, 255, 0.1)`.
*   **Modern Upload Buttons:** Large, dashed-border zones. When a file is hovered, the border should animate into a solid Cyber Blue line with a subtle pulse effect.
*   **Video Player Container:** Framed with a technical "HUD" overlay. Use JetBrains Mono for the timestamp and frame-counter overlays in the corners of the video.
*   **Image Grid:** For extracted frames, use a tight 8px gap. Detections within the grid are highlighted with a 2px Neon Green or Alert Red stroke.
*   **Status Chips:** Small, high-contrast pills. "Authentic" uses a Secondary Green background at 15% opacity with 100% opacity text.
*   **Inputs:** Dark backgrounds (`#000000`) with a 1px border. Focus state triggers a Cyber Blue glow and border transition.