/**
 * Sidebar navigation entries.
 *
 * ⚠ IN THEIR OWN MODULE so `Sidebar.tsx` exports only components. React Fast Refresh stops
 * working for a file that mixes the two, and the lint rule that says so is right — but the
 * list also needs to be importable, because `SidebarProbeNaming.test.tsx` asserts that no two
 * entries share a label. Rendering the whole sidebar to check a string would be a slower test
 * of a weaker property.
 */

import { Activity, Boxes, CircleHelp, FileJson, Layers, LayoutDashboard, MessagesSquare, Radar, Server, Settings, Share2, Sliders } from 'lucide-react';

export const navItems = [
  // Order mirrors miStudio's sidebar for the labels the two share
  // (Models → SAEs → … → Clusters → Circuits → Steering → Monitor), so moving
  // between the authoring tool and the serving runtime doesn't relearn the nav.
  // miLLM-only entries keep their relative position within that frame.
  { id: 'dashboard', label: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  { id: 'models', label: 'Models', path: '/models', icon: Server },
  { id: 'sae', label: 'SAEs', path: '/sae', icon: Layers },
  { id: 'profiles', label: 'Profiles', path: '/profiles', icon: FileJson },
  { id: 'clusters', label: 'Clusters', path: '/clusters', icon: Boxes },
  { id: 'circuits', label: 'Circuits', path: '/circuits', icon: Share2 },
  { id: 'steering', label: 'Steering', path: '/steering', icon: Sliders },
  // ⚠ RENAMED from "Probe" (D8). This page monitors SAE FEATURE activations; Feature 24's
  // "Probe Monitors" is a different thing entirely — a trained linear detector. Two pages both
  // called Probe would be a coin flip for an operator, and the URL is unchanged so no bookmark
  // or runbook breaks.
  { id: 'monitoring', label: 'Feature Monitor', path: '/monitoring', icon: Activity },
  { id: 'probe-monitors', label: 'Probe Monitors', path: '/probe-monitors', icon: Radar },
  // Feature 36: the conversations the armed probes saw, as timelines of behaviours.
  { id: 'sessions', label: 'Sessions', path: '/sessions', icon: MessagesSquare },
  // ENH-004 (suite layout, 2026-10-20): Help is the LAST main-list item, as in every suite app.
  // The footer keeps Settings, the book-icon manual link and the version.
  { id: 'help', label: 'Help', path: '/help', icon: CircleHelp },
];

export const bottomNavItems = [
  { id: 'settings', label: 'Settings', path: '/settings', icon: Settings },
];
