// In-app Help (ENH-004). One topic per screen, plus a first-run topic, a "when something fails"
// topic and a glossary, in plain language. This is the summary a user needs on screen; the public
// manual (config/manual.ts) is the full reference, and each topic names the manual page (and
// section) that explains it in depth.
//
// Origin: miDataworks frontend/src/config/help.ts @ 32cf406 (shape `HelpTopic`), extended with
// linkable topics (/help/<topic>) and HELP_FOR_PATH, which picks the topic the header links to.
//
// Guards (src/__tests__/help/):
//   manualLinks.test.tsx    every topic's manual page and #heading exists in manual/docs
//   plainLanguage.test.ts   no internal ids, and every acronym is spelled out at its first use
//   helpWiring.test.tsx     the nav entry, the route and the header link reach this content

export interface HelpTopic {
  id: string;
  title: string;
  /** Paragraphs; each is plain text. */
  body: string[];
  /** Short "how to" steps, in order. */
  steps?: string[];
  /** The manual page that explains this in depth, relative to the manual's base URL, with an
   *  optional "#section" (a heading slug on that page). */
  manual: string;
}

export const HELP_TOPICS: HelpTopic[] = [
  {
    id: 'getting-started',
    manual: 'getting-started/quickstart',
    title: 'Getting started',
    body: [
      'miLLM serves a language model and lets you look inside it while it answers. You can push its answers toward or away from an idea (steering), and watch which ideas light up (monitoring).',
      'Both work through a sparse autoencoder (SAE): a small network trained on one layer of the model that splits what that layer is doing into many separate features.',
    ],
    steps: [
      'Models: download a model from Hugging Face and load it.',
      'SAEs: download an SAE trained on that model and attach it, normally at the layer it was trained on.',
      'Steering: add a feature by its number, give it a strength, and switch steering on.',
      'Send a chat message (from Open WebUI, or any client of the OpenAI-compatible interface) and compare the answer with steering off.',
      'Feature Monitor: watch which features fire on your own prompts, to find the next ones to steer.',
    ],
  },
  {
    id: 'dashboard',
    manual: 'getting-started/dashboard#dashboard-overview',
    title: 'Dashboard',
    body: [
      'The dashboard shows at a glance whether a model is loaded, a sparse autoencoder (SAE) is attached, steering is on and the Feature Monitor is live, with the use, memory and temperature of the graphics cards.',
      'The Quick Start checklist walks through the first steps in order and links to each page. The bar along the top of every page repeats the essentials: the loaded model, the attached SAE and the connection to the server.',
    ],
  },
  {
    id: 'models',
    manual: 'features/model-management#loading--unloading',
    title: 'Models',
    body: [
      'Download a model from Hugging Face, then press Load. One model is served at a time: loading another unloads the current one first, and requests already running are given time to finish. Before loading, miLLM estimates the memory the model needs and warns when it will not fit.',
      'Several precisions of one model can be kept side by side; each has its own name, so a chat request always reaches the one it names.',
      'While a sparse autoencoder (SAE) is attached, the model is locked, so a chat request naming another model cannot replace it. A model leased by another tool, for example a labeling job in miDataworks, shows who holds it and until when; nobody can swap it out until the lease ends.',
    ],
  },
  {
    id: 'saes',
    manual: 'features/sae-management#attaching-an-sae',
    title: 'Sparse autoencoders (SAEs)',
    body: [
      'A sparse autoencoder (SAE) is trained on one layer of one model. Download it, then press Attach to hook it to the loaded model, normally at the layer it was trained on, which is the default.',
      'Attaching checks that it fits: a size that does not match the model is refused, while a different layer or model family only warns, because the features will mean less there. Attaching locks the model and turns on steering and the Feature Monitor.',
      'Detaching waits for running requests, clears every steering value and unlocks the model, so attaching again always starts clean.',
    ],
  },
  {
    id: 'profiles',
    manual: 'features/profiles',
    title: 'Profiles',
    body: [
      'A profile saves a set of features and their strengths, with the model and sparse autoencoder (SAE) they were made for, so you can switch the same steering on again later or share it.',
      'Save one from the Steering page with Save as Profile, or create one here. Activate replaces whatever steering is on with the profile\'s; it does not merge the two. A chat request can also name a profile to steer that one request only.',
    ],
  },
  {
    id: 'clusters',
    manual: 'features/clusters#activating--the-intensity-dial-λ',
    title: 'Clusters',
    body: [
      'A cluster is a group of features tuned together in miStudio, or shared on Hugging Face, and imported here as one steering definition. Import takes a pasted or uploaded definition file, or a pack found on Hugging Face.',
      'Activate applies every member at its authored strength, scaled by the intensity dial (λ). Turn the dial to make the whole cluster weaker or stronger within the range its author marked as safe. A cluster\'s own strengths cannot be edited here: use the dial, or import an updated definition.',
      'A cluster that does not match the attached sparse autoencoder (SAE) still imports, so you can read it, but it will not activate until a matching one is attached.',
    ],
  },
  {
    id: 'circuits',
    manual: 'features/circuits#serving-a-circuit',
    title: 'Circuits',
    body: [
      'A circuit is a set of features on several layers of the model, steered together as one. It is built in miStudio and imported here.',
      'Every circuit carries an evidence rung that says how well supported it is, and the page shows it in words. A circuit below rung 2 activates only after you acknowledge that.',
      'Activating attaches each sparse autoencoder (SAE) the circuit needs. When one cannot be attached, the circuit is served as a partial slice and the page says so. One intensity dial (λ) scales the whole circuit, and a chat request can set its own intensity for that request only.',
    ],
  },
  {
    id: 'steering',
    manual: 'features/feature-steering#strength-values',
    title: 'Steering and strength',
    body: [
      'Steering adds chosen features to the model\'s working state while it writes, which pushes the answer toward the idea each feature stands for. It needs a loaded model and an attached sparse autoencoder (SAE).',
      'Each feature has a strength. 0 does nothing; 1 to 10 is subtle; 10 to 50 brings the idea out clearly; above 100 the answer often turns repetitive or garbled. A negative strength suppresses the idea. The allowed range is -200 to +200.',
    ],
    steps: [
      'Add a feature by its number, or several at once with Batch Add (for example 1234:2.5).',
      'Start between 10 and 30, compare answers with steering on and off, and raise it gradually.',
      'Switch steering on to apply it to every request; Save as Profile keeps the set for later.',
    ],
  },
  {
    id: 'feature-monitor',
    manual: 'features/probe-monitoring',
    title: 'Feature Monitor',
    body: [
      'The Feature Monitor shows which features of the attached sparse autoencoder (SAE) fire while the model answers. It only watches: it never changes what the model says.',
      'Turn it on, choose how many of the strongest features to keep for each request, and if you like, list the feature numbers you want to watch and no others. Each record shows the features active as the model finished its answer, not a trace of every word.',
      'Pause freezes the display while recording carries on. This page is not Probe Monitors, which runs trained detectors from miStudio.',
    ],
  },
  {
    id: 'probe-monitors',
    manual: 'features/probe-monitors#arming-four-gates',
    title: 'Probe Monitors',
    body: [
      'A probe is a small detector trained in miStudio that scores chat requests for one property and gives a verdict when the score reaches its bar. Probes record and report; they never stop or change an answer.',
      'Arming a probe checks four things, in order, and a refusal names the one that failed: the limit on armed probes; that the loaded model is the one the probe was trained on; its evidence rung, where a probe below rung 2 needs your acknowledgement; and parity, which means this server reproduces the scores miStudio recorded.',
      'An armed probe that is not scoring always says why. Verdicts appear on this page, not in the chat.',
    ],
  },
  {
    id: 'sessions',
    manual: 'features/probe-monitors#sessions',
    title: 'Sessions',
    body: [
      'A session is one conversation, as the armed probes saw it. Each row is one turn and each column one behaviour, so you can see where a behaviour fired. Sessions are kept only while at least one probe is armed.',
      'A client can name its conversation with the X-miLLM-Session request header; in Open WebUI, add the header X-miLLM-Session with the value {{CHAT_ID}} to the miLLM connection. Without the header, miLLM recognises a conversation when a request continues an earlier one, and a conversation idle for 30 minutes ends.',
      'Recording text is off unless you turn it on in Settings. When on, the newest user message and the reply of each turn are kept for 7 days, never sent to the live display, and Show text fetches them only when you ask. Deleting a session removes its text at once.',
    ],
  },
  {
    id: 'approvals',
    manual: 'features/approvals#approving-and-rejecting',
    title: 'Approvals',
    body: [
      'When an artificial intelligence (AI) agent asks miLLM to do something that needs you, the request waits. The Approvals menu shows the number, and a banner shows it on every screen.',
      'Approve carries it out once, exactly as shown. Reject means it never runs. A request you do not decide expires after 24 hours.',
      'Each app decides its own requests. Requests an agent sends through miStudio to change miLLM wait here, in miLLM.',
      'History keeps every decision: who decided, when, the reason and what happened.',
    ],
  },
  {
    id: 'settings',
    manual: 'getting-started/dashboard#settings-page',
    title: 'Settings',
    body: [
      'Settings holds the theme, the connection to the server with a reconnect button, and the Hugging Face token.',
      'Save the token once and downloads and previews of gated or private repositories use it. Test asks Hugging Face which account it belongs to, and Clear removes it. A token typed into a download form still wins for that one download. The server stores the token encrypted, and refuses to save it until its encryption key is set; the page says so when that happens.',
      'The Environment section lists every setting the server was started with. About thirty of them can be changed while miLLM runs, such as the log level, the queue length and the Batch file limits: each row says what it does, when a change takes effect, the value from the environment beside the one in force, and who changed it last and why. Saving stores an override; Use the environment value removes it. Recent changes lists the last fifty.',
      'Settings read when a model loads or when the server starts are shown read-only, with what applying them takes. Secrets such as the database address are shown only as set or not set.',
      'The page also shows the version the server reports and the addresses it serves.',
    ],
  },
  {
    id: 'errors',
    manual: 'troubleshooting',
    title: 'When something fails',
    body: [
      'A refusal names what went wrong and what to try. Requests the server refuses carry an error code, and the manual lists every code with what to do about it.',
      'If steering seems to have no effect, check in this order: steering is switched on with strengths other than 0; the sparse autoencoder (SAE) is attached at the layer it was trained on; and the strength is large enough to show, so try 10 to 30 and then more.',
      'If a model will not load, it usually does not fit in the memory of the graphics card; try a smaller precision of the same model.',
    ],
  },
  {
    id: 'glossary',
    manual: 'concepts/interpretability#vocabulary',
    title: 'Words used here',
    body: [
      'Sparse autoencoder (SAE): a small network trained on one layer of a model that splits that layer\'s activity into many features, most of them off at any moment.',
      'Feature: one pattern an SAE learned, numbered from 0. Many features track an idea a person can recognise.',
      'Activation: how strongly a feature fires on a piece of text.',
      'Layer: one stage of the model\'s internal computation. An SAE is trained on one layer and is attached there.',
      'Steering: adding features to the model\'s working state while it writes. Strength says how much: 0 is none, and a negative strength suppresses the idea.',
      'Profile: a saved set of features and strengths you can switch on in one step.',
      'Cluster: features tuned together in miStudio and imported as one, with an intensity dial (λ) that scales them all.',
      'Circuit: features on several layers steered together as one, imported from miStudio.',
      'Evidence rung: a number from 0 to 3 saying how well supported a circuit or a probe is. Rung 0 means it was observed and nothing more; higher rungs passed stronger tests.',
      'Probe: a small trained detector that scores text for one property and gives a verdict when the score reaches its bar. Parity is the check that miLLM reproduces the scores miStudio recorded.',
    ],
  },
];

export const HELP_TOPIC_IDS = HELP_TOPICS.map((t) => t.id);

/** The topic with this id, if there is one. */
export function helpTopic(id: string | undefined): HelpTopic | undefined {
  return id ? HELP_TOPICS.find((t) => t.id === id) : undefined;
}

/** The Help topic the header links to on each screen, by route path. */
export const HELP_FOR_PATH: Record<string, string> = {
  '/dashboard': 'dashboard',
  '/models': 'models',
  '/sae': 'saes',
  '/profiles': 'profiles',
  '/clusters': 'clusters',
  '/circuits': 'circuits',
  '/steering': 'steering',
  '/monitoring': 'feature-monitor',
  '/probe-monitors': 'probe-monitors',
  '/sessions': 'sessions',
  '/settings': 'settings',
  '/approvals': 'approvals',
};

/** The Help topic for a location, or undefined (the Help page itself, an unknown path). */
export function helpTopicForPath(pathname: string): string | undefined {
  if (pathname === '/') return HELP_FOR_PATH['/dashboard'];
  return HELP_FOR_PATH[pathname.replace(/\/+$/, '')];
}

/** The in-app address of a topic: linkable and bookmarkable. */
export function helpPath(topic?: string): string {
  return topic ? `/help/${encodeURIComponent(topic)}` : '/help';
}
