"""Teacher-input data adapter for native ReTRACE tool-ending conversations.

Only teacher context composition is specialized. Official TRL SDFT sampling,
loss, optimization, checkpointing, and student inputs are not replaced.
Pinned to TRL 1.13.0's experimental context-builder interface; not deployed in
any training run until its real-input integration test passes.
"""
import copy


def compose_teacher_prompt(prompt, privileged_context, template):
    if not isinstance(prompt, list) or not isinstance(privileged_context, str) or not privileged_context.strip():
        raise ValueError('native conversation and nonempty reference required')
    result = copy.deepcopy(prompt)
    user = next((m for m in result if m.get('role') == 'user'), None)
    if user is None or not isinstance(user.get('content'), str):
        raise ValueError('first task user message must be textual')
    user['content'] = template.format(prompt=user['content'], privileged_context=privileged_context)
    return result


def make_teacher_context_builder(trainer):
    from trl.experimental.sdft.sdft_trainer import DemonstrationTeacherContextBuilder

    class ReTraceTeacherContextBuilder(DemonstrationTeacherContextBuilder):
        def _compose_teacher_prompt(self, prompt, privileged_context):
            return compose_teacher_prompt(prompt, privileged_context, self.trainer.args.teacher_prompt_template)

    return ReTraceTeacherContextBuilder(trainer)
