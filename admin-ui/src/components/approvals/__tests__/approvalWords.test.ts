/**
 * T28 (040 FR-040.32): History and the cards name the agent, whichever MCP server filed the request.
 * Rows stored before the rename carry the old MCP actor id (DR-17); rows filed after it carry the
 * new one. Both read as an agent, without the `agent:` prefix.
 *
 * Control NC-86 (0xcc/reviews/040_P4_controls_2026-11-11.md): make `agentName` keep the prefix ->
 * "names both agent ids without the prefix" red.
 */
import { describe, expect, it } from 'vitest';

import { agentName, askedBy } from '../approvalWords';

const OLD = ['agent', 'mistudio-mcp'].join(':');
const NEW = 'agent:miworkbench-mcp';

describe('approvalWords', () => {
  it('names both agent ids without the prefix', () => {
    expect(agentName(OLD)).toBe('mistudio-mcp');
    expect(agentName(NEW)).toBe('miworkbench-mcp');
    expect(askedBy(OLD)).toBe('asked by agent mistudio-mcp');
    expect(askedBy(NEW)).toBe('asked by agent miworkbench-mcp');
  });

  it('says "an agent" when no name was recorded', () => {
    for (const blank of [null, undefined, '', 'agent', 'agent:']) {
      expect(agentName(blank)).toBe('an agent');
      expect(askedBy(blank)).toBe('asked by an agent');
    }
  });
});
