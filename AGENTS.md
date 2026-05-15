# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

BetterDiscordApp is a client-side modification for the Discord desktop app. It is **not** a standalone web service — it injects JS/CSS into Discord's Electron renderer. There are no backend servers, databases, or APIs to run.

Two build targets exist side-by-side:

| Target | Location | Build command | Notes |
|--------|----------|---------------|-------|
| **v1** | `/workspace` | `grunt concat uglify cssmin` | Production-ready. Skip the `sass` task (requires Ruby); pre-built CSS exists in `dev/css/main.css`. |
| **v2** | `/workspace/v2` | `grunt requirejs babel replace:nongreedy` | Preview/experimental. The `amdclean` step fails because `grunt-amdclean` is no longer published on npm; run the available tasks individually. |

### Building

- **v1 full build (without sass):** `cd /workspace && grunt concat uglify cssmin`
- **v2 partial build:** `cd /workspace/v2 && grunt requirejs babel replace:nongreedy`
- The root `grunt` default task includes `sass` which requires Ruby + the `sass` gem. Use the explicit task list above to skip it.

### Known limitations

- `grunt-amdclean` has been removed from npm. The v2 `amdclean` and `js` tasks will fail. Use the individual tasks (`requirejs`, `babel`, `replace:nongreedy`) instead.
- `grunt-contrib-sass` requires Ruby and the `sass` gem. The pre-compiled CSS in `dev/css/main.css` is checked in, so sass compilation is only needed when editing `.sass` source files.
- There are no automated tests or lint configurations in this repository.
- This is a Discord client mod — full end-to-end testing requires the Discord desktop app, which is not available in the cloud VM. Build verification is the primary validation method.

### v2 dependency workaround

Because `grunt-amdclean` is unpublished, `npm install` in `v2/` fails. The update script works around this by temporarily swapping `package.json` to install only the available packages. See the update script for details.
