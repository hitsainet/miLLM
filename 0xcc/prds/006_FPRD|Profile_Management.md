# Feature PRD: Profile Management

## miLLM Feature 6

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**Feature Priority:** Secondary (P2)
**References:**
- Project PRD: `000_PPRD|miLLM.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Feature Overview

### Feature Name
**Profile Management** - Save, load, and manage steering configuration profiles.

### Brief Description
Profile Management enables users to save steering configurations as named profiles, making it easy to switch between different behavioral presets and share configurations. Profiles persist steering values along with metadata about the associated model and SAE.

### Problem Statement
Users experimenting with steering need to:
- Save successful configurations for reuse
- Switch between different presets quickly
- Share configurations with others
- Reproduce specific steering setups reliably

### Feature Goals
1. **Persistence:** Save steering configurations with names and descriptions
2. **Quick Activation:** Single-click profile activation
3. **API Access:** Select profiles via API parameter
4. **Portability:** Export/import for sharing and miStudio compatibility
5. **Organization:** List, edit, delete profiles

### User Value Proposition
Users can save their steering experiments as profiles, enabling quick recall of successful configurations and easy sharing with others. This transforms one-time experiments into reusable presets.

### Connection to Project Objectives
- **FR-6.1 through FR-6.5:** Direct implementation requirements

---

## 2. User Stories & Scenarios

### Primary User Stories

#### US-6.1: Create Profile from Current Settings
**As a** user who found a good steering configuration
**I want to** save it as a named profile
**So that** I can reuse it later

**Acceptance Criteria:**
- [ ] Click "Save as Profile" button
- [ ] Enter profile name and optional description
- [ ] Current steering values saved
- [ ] Model and SAE IDs recorded
- [ ] Profile appears in list

#### US-6.2: Activate a Profile
**As a** user switching between configurations
**I want to** activate a saved profile
**So that** its steering values are applied

**Acceptance Criteria:**
- [ ] Select profile from list
- [ ] Click "Activate" button
- [ ] Steering values loaded and applied
- [ ] Steering automatically enabled
- [ ] UI reflects active profile

#### US-6.3: Select Profile via API
**As a** developer integrating with miLLM
**I want to** specify a profile in API requests
**So that** I can control steering programmatically

**Acceptance Criteria:**
- [ ] Add `profile` parameter to chat/completion requests
- [ ] Profile steering applied for that request
- [ ] Does not change global active profile
- [ ] Error if profile not found

#### US-6.4: Export Profile
**As a** user sharing configurations
**I want to** export a profile as JSON
**So that** others can import it

**Acceptance Criteria:**
- [ ] Export button generates JSON file
- [ ] JSON includes all steering values
- [ ] JSON includes model/SAE metadata
- [ ] Format compatible with miStudio

#### US-6.5: Import Profile
**As a** user receiving a shared profile
**I want to** import it
**So that** I can use the same configuration

**Acceptance Criteria:**
- [ ] Import accepts JSON file
- [ ] Validates format before import
- [ ] Warns if model/SAE mismatch
- [ ] Profile added to list

### Edge Cases

#### EC-6.1: Profile with Mismatched SAE
- **Trigger:** Activate profile for different SAE
- **Behavior:** Show warning, offer to proceed or cancel
- **Message:** "Profile was created for different SAE. Feature indices may not match."

#### EC-6.2: Duplicate Profile Name
- **Trigger:** Create profile with existing name
- **Behavior:** Offer to overwrite or rename
- **Message:** "Profile 'name' already exists. Overwrite?"

---

## 3. Functional Requirements

### Profile Persistence (FR-6.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| PRF-P1 | System shall save steering configurations as profiles | Must |
| PRF-P2 | Profiles shall have unique names | Must |
| PRF-P3 | Profiles shall store steering values as feature-value pairs | Must |
| PRF-P4 | Profiles shall record associated model and SAE IDs | Must |
| PRF-P5 | Profiles shall support optional description | Should |

### Profile Selection (FR-6.2, FR-6.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| PRF-S1 | System shall allow profile selection via admin UI | Must |
| PRF-S2 | System shall allow profile selection via API parameter | Must |
| PRF-S3 | System shall apply profile steering on activation | Must |
| PRF-S4 | System shall enable steering when profile activated | Must |

### Import/Export (FR-6.4, FR-6.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| PRF-E1 | System shall export profiles as JSON | Must |
| PRF-E2 | System shall import profiles from JSON | Must |
| PRF-E3 | System shall validate import format | Must |
| PRF-E4 | Export format shall be miStudio-compatible | Should |

### Input/Output Specifications

#### Profile Schema
```typescript
interface Profile {
  id: string;                     // Unique identifier
  name: string;                   // Display name
  description?: string;           // Optional description
  model_id: string;               // Associated model
  sae_id: string;                 // Associated SAE
  steering: Record<number, number>;  // Feature index → value
  created_at: string;             // ISO timestamp
  updated_at: string;             // ISO timestamp
}
```

#### Export Format (miStudio-compatible)
```json
{
  "version": "1.0",
  "type": "millm_profile",
  "profile": {
    "name": "Yelling Demo",
    "description": "Makes model respond in caps",
    "model": "google/gemma-2-2b",
    "sae": "jbloom/gemma-2-2b-res-jb",
    "layer": 12,
    "steering": {
      "1234": 5.0,
      "5678": -3.0
    }
  },
  "exported_at": "2026-01-30T12:00:00Z",
  "exported_from": "miLLM v1.0"
}
```

---

## 4. Data Requirements

### Database Schema

```sql
CREATE TABLE profiles (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    model_id VARCHAR(100),
    sae_id VARCHAR(50),
    layer INTEGER,
    steering JSONB NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_profiles_name ON profiles(name);
CREATE UNIQUE INDEX idx_active_profile ON profiles(is_active) WHERE is_active = TRUE;
```

---

## 5. API Specifications

### Management API Endpoints

#### GET /api/profiles
List all profiles.
```json
Response: {
  "profiles": [
    {
      "id": "prf-abc123",
      "name": "Yelling Demo",
      "description": "...",
      "model_id": "...",
      "sae_id": "...",
      "feature_count": 3,
      "is_active": true,
      "created_at": "..."
    }
  ],
  "active_profile_id": "prf-abc123"
}
```

#### POST /api/profiles
Create new profile.
```json
Request: {
  "name": "Yelling Demo",
  "description": "Makes model yell",
  "steering": {"1234": 5.0}
}

Response: {
  "id": "prf-abc123",
  "name": "Yelling Demo",
  ...
}
```

#### POST /api/profiles/{id}/activate
Activate a profile.

#### GET /api/profiles/{id}/export
Export profile as JSON file.

#### POST /api/profiles/import
Import profile from JSON.

---

## 6. Success Criteria

### Completion Criteria
- [ ] Can create profiles from current steering
- [ ] Can activate profiles
- [ ] Can select profiles via API
- [ ] Can export/import profiles
- [ ] Only one profile active at a time

---

**Document Status:** Complete
**Next Document:** `006_FTDD|Profile_Management.md`
