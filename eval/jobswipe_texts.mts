// Build the exact strings JobSwipe embeds, using JobSwipe's own TypeScript.
//
// Nothing here re-implements JobSwipe logic: buildProfileTexts / buildJobText /
// stripContactInfo / chunkText come from supabase/functions/_shared/embeddings.ts and
// extractFromResume from src/lib/resumeExtract.ts, imported unchanged from the cloned
// repo. Node >= 22.18 strips the TypeScript types natively.
//
//   node eval/jobswipe_texts.mts <in.json> <out.json>
//
// in.json:  {"resumes": [str, ...], "jobs": [str, ...]}
// out.json: {"profiles_suggested": [{header, chunks}], "profiles_resume_only": [...],
//            "jobs": [str, ...]}
import { readFileSync, writeFileSync } from 'node:fs';
import { buildJobText, buildProfileTexts } from '../repos/jobswipe/supabase/functions/_shared/embeddings.ts';
import { extractFromResume } from '../repos/jobswipe/src/lib/resumeExtract.ts';

const [inPath, outPath] = process.argv.slice(2);
const { resumes, jobs } = JSON.parse(readFileSync(inPath, 'utf8')) as { resumes: string[]; jobs: string[] };

// "Suggested" profile: the onboarding flow where the seeker uploads a resume and keeps
// every suggestion the "From your resume" card offers (title, skills, summary -> bio).
const profiles_suggested = resumes.map((text) => {
  const s = extractFromResume(text);
  return buildProfileTexts({ target_title: s.title, skills: s.skills, bio: s.summary, resume_text: text });
});

// Resume-only profile: the seeker uploads a resume and accepts nothing else.
const profiles_resume_only = resumes.map((text) => buildProfileTexts({ resume_text: text }));

// The labelled pairs carry only a description, no title / skills / location.
// The title is unknown, so it is passed as '' and filtered out by compact() — the JD's own
// first line usually names the role anyway.
const jobTexts = jobs.map((description) => buildJobText({ title: '', description }).replace(/^Role: /, ''));

writeFileSync(outPath, JSON.stringify({ profiles_suggested, profiles_resume_only, jobs: jobTexts }));
console.log(`built ${resumes.length} profiles and ${jobs.length} job texts`);
