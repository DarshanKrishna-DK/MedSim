/**
 * Branch guard and MedSim-Gemini governance verification script.
 *
 * Checks:
 * 1. Required project rule files (GEMINI.md, .agents/rules/medsim-gemini.md) and Cloud Run config (Dockerfile, cloudbuild.yaml) exist.
 * 2. Git branch check: if .git exists, current branch is MedSim-Gemini (skips cleanly with PASS if in CI/Docker without .git).
 * 3. LLM Effort Cap check: no backend code hardcodes "reasoning_effort": "high" or "thinking_level": "HIGH".
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

type Violation = { case: string; rule: string; detail: string };

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export function verifyBranchGuard(): Violation[] {
  const violations: Violation[] = [];
  const rootDir = path.resolve(__dirname, '../..');

  // 1. Check required project files
  const requiredFiles = [
    'GEMINI.md',
    '.agents/rules/medsim-gemini.md',
    'Dockerfile',
    'cloudbuild.yaml',
  ];

  for (const rel of requiredFiles) {
    const fullPath = path.join(rootDir, rel);
    if (!fs.existsSync(fullPath)) {
      violations.push({
        case: 'branch-guard',
        rule: 'RequiredFileMissing',
        detail: `Required governance or deployment file missing: ${rel}`,
      });
    }
  }

  // 2. Git branch check
  const gitDir = path.join(rootDir, '.git');
  if (fs.existsSync(gitDir)) {
    try {
      const currentBranch = execSync('git branch --show-current', {
        cwd: rootDir,
        encoding: 'utf-8',
      }).trim();

      if (currentBranch && currentBranch !== 'MedSim-Gemini') {
        violations.push({
          case: 'branch-guard',
          rule: 'InvalidBranch',
          detail: `Current git branch is '${currentBranch}'. All work must be on 'MedSim-Gemini'.`,
        });
      }
    } catch (err) {
      // Ignore git exec errors in restricted sandbox environments
    }
  }

  // 3. LLM Effort Cap check (backend files)
  const backendDir = path.join(rootDir, 'backend');
  if (fs.existsSync(backendDir)) {
    const pyFiles: string[] = [];
    function walk(dir: string) {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          if (entry.name !== '__pycache__' && entry.name !== '.venv') {
            walk(full);
          }
        } else if (entry.name.endsWith('.py')) {
          pyFiles.push(full);
        }
      }
    }
    walk(backendDir);

    for (const file of pyFiles) {
      const content = fs.readFileSync(file, 'utf-8');
      if (
        content.includes('"reasoning_effort": "high"') ||
        content.includes("'reasoning_effort': 'high'") ||
        content.includes('"thinking_level": "HIGH"') ||
        content.includes("'thinking_level': 'HIGH'")
      ) {
        const relPath = path.relative(rootDir, file);
        violations.push({
          case: 'branch-guard',
          rule: 'ForbiddenHighEffort',
          detail: `File ${relPath} contains prohibited high reasoning effort / thinking level. Effort must be capped at low or medium.`,
        });
      }
    }
  }

  return violations;
}
