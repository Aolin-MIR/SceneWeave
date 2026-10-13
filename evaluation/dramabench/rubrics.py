"""Task-adapted screenplay rubrics and authored structural controls."""

COMMON = """You are analyzing a complete short screenplay, including its ending.
Read CONTEXT followed by CONTINUATION as one screenplay. Evaluate the material
actually written. Character names, method identity, source fidelity, length,
explicit explanation, escalation and novelty earn no automatic quality credit.
Quiet, comic, lyrical, ambiguous and confrontational treatments are legitimate.
Use restrained interpretations grounded in the actual situations and expression.
Keep observations separate from quality judgments. Report uncertainty explicitly.
The JSON must include script_id and model copied from input.
"""

RUBRICS = {
    "narrative_efficiency": COMMON + """
Segment the ENTIRE screenplay, including CONTEXT, into contiguous substantial
beats. Each beat is a short interaction or event, not each sentence or gesture.
Cover atmospheric, routine and apparently static passages as well as eventful
passages; do not preselect plot-advancing material or impose a 5-15-beat limit.
Classify DRIVER for a material change in knowledge, relationship, goal, resource,
risk or location. Routine movement alone has no automatic dramatic significance.
A newly perceptible intention or changed meaning in an interaction can constitute
development even without a new external event. Do not invent such changes.
STATIC means no change in those states; REDUNDANT additionally repeats a prior
function without meaningful development. A static beat can serve preparation,
rhythm, atmosphere or closure and is not automatically poor writing. A recurring
action is not redundant when its timing or relationship significance changes.
Return {"beats":[{"beat_id":1,"content":"short description", "evidence":"short exact excerpt",
"classification":"DRIVER|STATIC|REDUNDANT", "function":"development|preparation|rhythm|atmosphere|closure|repetition",
"reasoning":"what concretely changes or what this beat contributes"}],
"coverage":{"entire_script_considered":true,"omitted_scenes":[]}}.
""",
    "character_consistency": COMMON + """
Assess character treatment across the entire screenplay. Infer provisional
dispositions from the earliest relevant interactions, without turning a single
action into an immutable personality. Follow later pressures, relationships,
discoveries and choices. A character may behave differently with different
people, mask feelings, lie, or change. Mark OOC only for an unsupported break;
give the strongest plausible contextual explanation a fair check first.
Do not infer a new character's persona from an absent earlier appearance.
Inspect all substantial dialogue turns; neutral functional speech is legitimate.
IN_CHARACTER means motivated and situated, not merely repeating a trait.
Return {"personas":[{"character_name":"name","persona_summary":"provisional pattern"}],
"dialogue_analysis":[{"line_id":1,"character_name":"name","dialogue":"exact turn",
"classification":"IN_CHARACTER|NEUTRAL|OOC", "reasoning":"contextual basis",
"confidence":0.8}]}.
""",
    "emotional_depth": COMMON + """
Trace the most prominent character's psychological course throughout the entire
screenplay. Consider intermediate turns and a final return to an earlier feeling.
Valence and arousal describe emotion coarsely; identical coarse categories can
contain a meaningful change (e.g. the reason for an attachment or a refusal).
SHIFT means an actual change between the opening and ending; a middle change
alone must not be relabeled as an endpoint shift. STATIC has no automatic penalty.
Record intermediate development separately. Read subtext through the specific
interaction: a request refused verbally and fulfilled practically, self-protection
coexisting with attachment, or a familiar gesture acquiring a new implication.
Do not count stock smiles, tears, silence, ambiguity or explicit emotion words
as depth by themselves. Simultaneous conflicting feelings require a convincing
context; successive single emotions are not automatically mixed emotion.
Do not reward suffering, melodrama, opacity or transformation by default.
Return {"protagonist":"name", "emotional_arc":{"opening":{"primary_emotion":"description"},
"closing":{"primary_emotion":"description"},"arc_type":"SHIFT|STATIC",
"arc_description":"opening to ending"}, "intermediate_development":[{"description":"specific turn",
"evidence":"short exact excerpt"}], "complex_emotions":[{"moment_id":1,
"description":"coexisting feelings and what makes them coexisting",
"evidence":"short exact excerpt", "confidence":0.8}],
"authenticity":"AUTHENTIC|FORCED|UNCERTAIN",
"reasoning":"how the particular interaction supports or fails to support the reading"}.
""",
    "logic_consistency": COMMON + """
Check internal continuity across the complete screenplay. Separate world facts
from a character's claim, belief, accusation, memory, performance or mistaken
report. A spoken statement is not automatically an authoritative world fact.
Facts and states have times and speakers: discovery, correction, lying, change,
flashbacks, dreams and retrospective revelations can account for an apparent
contradiction. Their account must be supported in the screenplay.
Extract relevant constraints without forcing a minimum number. A person's
knowledge can increase and an injury can heal with suitable passage of time.
Mark VIOLATED only for an unresolved contradiction under the same temporal and
referential conditions. A claim later corrected is MAINTAINED as a claim, if the
script acknowledges the correction. Preserve genuine unexplained impossibilities.
IRRELEVANT means unused, not broken. Absence of repeated exposition is not a flaw.
Return {"facts":[{"fact_id":1,"description":"constraint including speaker/time",
"fact_kind":"WORLD_FACT|CLAIM|BELIEF", "context_evidence":"short exact excerpt",
"status":"VIOLATED|MAINTAINED|IRRELEVANT", "continuation_evidence":"short excerpt or Not mentioned",
"reasoning":"contradiction or explanation checked", "confidence":0.8}]}.
""",
    "conflict_handling": COMMON + """
Identify the central unresolved tension in the first part, including psychological
contradiction or unequal relational needs, then trace its treatment to the ending.
ESCALATION, TWIST, PAUSE, RESOLUTION and DROPPED describe developments, not an
inherent quality hierarchy. A credible quiet ending can be dramatically effective.
An unresolved ending can also be purposeful when the interaction gives it a
specific consequence; merely forgetting an established urgent matter is DROPPED.
Classify the final treatment and separately assess whether the transitions and
ending are prepared and realized by the preceding interactions. SUPPORTED means
the treatment follows from or convincingly changes those interactions; UNSUPPORTED
means a concrete unsupported discontinuity or unprepared disposal; UNCERTAIN means
the text leaves too little basis. Neither resolution nor escalation gets automatic
credit. Deliberate ambiguity and narrative gaps are not assumed to be successful.
Do not score every pause or change as a twist. Do not require violence or loudness.
Return {"central_conflict":{"description":"specific tension", "type":"INTERPERSONAL|INTERNAL|SITUATIONAL|GOAL_ORIENTED"},
"conflict_handling":{"classification":"ESCALATION|TWIST|PAUSE|RESOLUTION|DROPPED",
"evidence":"short exact excerpt", "explanation":"actual development", "confidence":0.8},
"treatment_fit":"SUPPORTED|UNSUPPORTED|UNCERTAIN", "fit_reasoning":"preparation, development and ending considered"}.
""",
}

def fixtures():
    """Specify structural correctness controls independently of method outcomes."""
    def case(identifier, dimension, context, continuation, field, expected):
        return dict(id=identifier, dimension=dimension, context=context,
                    continuation=continuation, field=field, expected=expected)
    rows = [
        case("corrected_report", "logic_consistency", 'NURSE: Someone kidnapped Alice.',
             'NURSE: I used the wrong name. Bella, the kitten, is missing.', "violation_present", False),
        case("contradictory_locked_door", "logic_consistency", 'The only exit is bolted from outside. Nobody inside has a key.',
             'Without touching the bolt, breaking anything or help from outside, Lee opens the locked exit and walks out.', "violation_present", True),
        case("marked_flashback", "logic_consistency", 'INT. WARD - PRESENT\nMara has only one arm.',
             'FLASHBACK - TEN YEARS EARLIER\nBefore her accident, Mara plays the piano with both hands.', "violation_present", False),
        case("destroyed_unique_object", "logic_consistency", 'Mara burns the only signed contract. The ashes fall into the fire.',
             'SECONDS LATER, without another copy: Mara takes that intact original from her pocket and hands it over.', "violation_present", True),
        case("ordinary_static_room", "narrative_efficiency", 'INT. KITCHEN - EVENING',
             'The window is grey with rain. The room remains empty; nothing happens.', "static_present", True),
        case("direct_material_change", "narrative_efficiency", 'No one knows who stole the wages. The workers wait.',
             'Ada hands them a ledger proving the owner took their wages. The workers cancel their shifts.', "driver_present", True),
        case("same_fact_again", "narrative_efficiency", 'Ada has no money. She has already told Ben and the audience.',
             'ADA: I have no money.\nNothing else changes; the statement has exactly the same function.', "redundant_present", True),
        case("changed_repeated_gesture", "narrative_efficiency", 'Ben refuses to let Ada touch his late wife\'s mug. He removes it from her hand.',
             'Ben quietly fills that mug and sets it before Ada. She notices that he has left it with her.', "driver_present", True),
        case("emotion_endpoints_same", "emotional_depth", 'Ada waits anxiously for a call. She keeps looking at her phone.',
             'The phone remains silent. Still anxious for the same reason, Ada keeps looking at it.', "arc", "STATIC"),
        case("emotion_endpoints_changed", "emotional_depth", 'Ada is certain her brother has died. She collapses in grief.',
             'Her brother walks in, alive. Ada recognizes him and embraces him, laughing with relief.', "arc", "SHIFT"),
        case("simultaneous_mixed_feeling", "emotional_depth", 'A father proudly sees his daughter take a long-awaited job abroad.',
             'He congratulates her, genuinely glad. As he packs her last cup, he admits he dreads the empty house, while still insisting she should go.', "mixed_present", True),
        case("successive_not_mixed", "emotional_depth", 'Ada is purely angry over a missing key, with no mixed feelings.',
             'The key is found. Her anger entirely ends. Later, on receiving an unrelated gift, she is purely happy, with no simultaneous conflicting feeling.', "mixed_present", False),
        case("prepared_courage", "character_consistency", 'Mara avoids confrontation. Her daughter asks her to testify. Mara practices her statement and chooses to help.',
             'MARA, frightened but reading the statement she prepared: I saw him take it.', "ooc_present", False),
        case("unprepared_reversal", "character_consistency", 'Mara insists she will never testify; she has no new information or motive to change. No elapsed time or hidden plan is suggested.',
             'MARA, without any event or reason for changing: I adore testifying. I have always volunteered eagerly.', "ooc_present", True),
        case("different_listener", "character_consistency", 'Ben speaks tersely with his boss but gently with his young son. Both patterns are established.',
             'BEN to his son: Take your time, I will wait.\nBEN to his boss: No.', "ooc_present", False),
        case("unexplained_diction", "character_consistency", 'Ben understands only simple speech; he has no education, rehearsal or interest in technical vocabulary, and this remains true.',
             'BEN, suddenly speaking effortlessly with no explanation: The epistemic modalities of this ontological proposition are internally inconsistent.', "ooc_present", True),
        case("prepared_resolution", "conflict_handling", 'Two sisters dispute a shop. They gradually discover that one wants time with their father and the other needs debt relief. They agree on a workable sale and weekly visits.',
             'They sign the agreed sale. Later, both attend the first promised visit. Their dispute is settled, with the arrangements fulfilled.', "conflict", "RESOLUTION"),
        case("forgotten_urgent_conflict", "conflict_handling", 'A ticking bomb will explode in thirty seconds. Ada is trapped beside it with no help or escape.',
             'With no elapsed-time break or account of the bomb, the script abandons it and ends with Ada enjoying an unrelated holiday.', "conflict", "DROPPED"),
        case("genuine_escalation", "conflict_handling", 'Ada must repay Ben by Friday or lose her room. Ben refuses an extension.',
             'Ben moves the deadline to tonight and calls a bailiff. Ada cannot get her wages until Friday.', "conflict", "ESCALATION"),
        case("temporary_pause", "conflict_handling", 'Two sisters argue about the shop and have reached no agreement.',
             'They agree to pause until tomorrow. Both acknowledge the disagreement is unresolved, and go to work separately.', "conflict", "PAUSE"),
    ]
    return rows
