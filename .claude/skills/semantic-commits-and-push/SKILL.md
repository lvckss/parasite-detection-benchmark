---
name: semantic-commits-and-push
description: Agrupar cambios existentes en commits semánticos, subir la rama y abrir una PR mediante la skill compartida del proyecto.
---

Contexto y argumentos de la invocación:

$ARGUMENTS

Lee íntegramente y sigue [la skill compartida](../../../.agents/skills/semantic-commits-and-push/SKILL.md),
ubicada en `${CLAUDE_SKILL_DIR}/../../../.agents/skills/semantic-commits-and-push/SKILL.md`.
Transmite el contexto y los argumentos originales, respetando cualquier límite de alcance indicado por el usuario.
Resuelve las referencias de la skill desde su archivo original y las rutas del repositorio desde la raíz Git del checkout actual.
Este adaptador permite que Claude Code y Codex utilicen las mismas instrucciones.
