# Notes

When I started learning robotics, I was quickly overwhelmed by the amount of information I *didn't* know, from machine learning topics like computer vision and reinforcement learning to classical robotics topics like kinematics and controls. The notes in the repository are constantly updated with papers, talks, courses, and more.

I've created simple note-taking system to streamline my learning. It uses Obsidian, Zotero, and GitHub for (free) syncing. I've included instructions below for setting up these notes locally and using the system for your own notes.

## Setup

1. Download [Obsidian](https://obsidian.md/) and [Zotero](https://www.zotero.org/).
2. Install [Zotero Connect](https://www.zotero.org/download/connectors), which lets you save papers to Zotero directly from the browser.
3. Install [Better BibTeX for Zotero](https://retorque.re/zotero-better-bibtex/installation/index.html), which facilitates citations in Obsidian using Zotero.
4. Clone this repository. The `.obsidian` directory contains some default Obsidian workspace settings and plugins.
5. Open Obsidian and load a new vault from this repository.
6. Setup citations in Obsidian from Zotero. Go to File -> Export Library. Set the format to *Better BibLaTex* and check *keepUpdated*. Click OK, then change the filename to `bib.bib` at the root of the notes directory. The Citations plugin in Obsidian will read from this file (If not, go to Settings -> Community Plugins -> Citations) and modify *Citation database format* to "BibLaTeX" and *Citation database path* to "bib.bib".

### Private Notes

You may have some information you don't want on a public GitHub repo. But creating a new private vault means you can't use Obsidian's excellent [linking](https://help.obsidian.md/Getting+started/Link+notes) functionality.

Instead, initialize a separate, private git repository called `personal-notes` at the root of this repo. This is gitignored by the main repo. The `sync.sh` script syncs both repos.

## Using These Notes

### Syncing

Open a terminal and go to `notes`. Your `personal-notes` should be a subdirectory. Running `./sync.sh` will sync both repos to GitHub.

**Important**: run `./sync.sh` *before* and *after* each note-taking session, especially if you are syncing across multiple devices.

### Adding Papers

I use Zotero Connect to download papers onto Zotero. Zotero is good for marking up PDFs.

If you set up citations in Obsidian from Zotero correctly, you can go to the command palette and select *Citations: Open literature note* and select the paper. This generates a nice template to fill out.

### Other Notes

The `templates` directory contains templates for other types of notes. There is also a paper template if you don't want to use Zotero. The `README` template is like a TLDR table of contents to link your notes to.
