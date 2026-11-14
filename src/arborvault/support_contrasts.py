"""Counterfactual support records; fictional facts, explicit four-way labels.

Each family belongs to exactly one role. Two questions cross two contexts so that
neither a question alone nor a context alone predicts support. These are authored
synthetic diagnostics, not independently adjudicated human questions.
"""

from .support_data import digest

NEGATION = [
    (
        ["{a} is allowed to {action}.", "{a} is not allowed to {action}."],
        ["Which unit is allowed to {action}?", "Which unit is not allowed to {action}?"],
    ),
    (
        [
            "The release policy says that {a} may {action}.",
            "The release policy says that {a} may not {action}.",
        ],
        [
            "Which unit may {action} under the release policy?",
            "Which unit may not {action} under the release policy?",
        ],
    ),
    (
        [
            "The register explicitly authorizes {a} to {action}.",
            "The register explicitly forbids {a} to {action}.",
        ],
        [
            "Which unit does the register authorize to {action}?",
            "Which unit does the register forbid to {action}?",
        ],
    ),
    (
        ["{a} has permission to {action}.", "{a} does not have permission to {action}."],
        ["Which unit has permission to {action}?", "Which unit lacks permission to {action}?"],
    ),
    (
        [
            "Review outcome: {a} is permitted to {action}.",
            "Review outcome: {a} is barred from {gerund}.",
        ],
        ["Which unit is permitted to {action}?", "Which unit is barred from {gerund}?"],
    ),
    (
        ["The audit records that {a} can {action}.", "The audit records that {a} cannot {action}."],
        [
            "According to the audit, which unit can {action}?",
            "According to the audit, which unit cannot {action}?",
        ],
    ),
    (
        [
            "A signed decision grants {a} permission to {action}.",
            "A signed decision denies {a} permission to {action}.",
        ],
        [
            "Which unit is granted permission to {action} by the signed decision?",
            "Which unit is denied permission to {action} by the signed decision?",
        ],
    ),
    (
        [
            "The request from {a} to {action} was approved.",
            "The request from {a} to {action} was rejected.",
        ],
        [
            "Which unit's request to {action} was approved?",
            "Which unit's request to {action} was rejected?",
        ],
    ),
    (
        [
            "The gatekeeper gave {a} authorization to {action}.",
            "The gatekeeper did not give {a} authorization to {action}.",
        ],
        [
            "To which unit did the gatekeeper give authorization to {action}?",
            "To which unit did the gatekeeper not give authorization to {action}?",
        ],
    ),
    (
        ["{a} is cleared to {action}.", "{a} is not cleared to {action}."],
        ["Which unit has clearance to {action}?", "Which unit has no clearance to {action}?"],
    ),
    (
        [
            "Permission to {action}: {a} is authorized.",
            "Permission to {action}: {a} is unauthorized.",
        ],
        ["Name the unit authorized to {action}.", "Name the unit unauthorized to {action}."],
    ),
    (
        ["For {a}, the policy allows {gerund}.", "For {a}, the policy prohibits {gerund}."],
        [
            "Under this policy, which unit may {action}?",
            "Under this policy, which unit may not {action}?",
        ],
    ),
]

RELATION = [
    (
        "{sender} shipped {asset} to {receiver}; the dispatch reference was {code}.",
        "What dispatch reference belongs to {asset} sent from {sender} to {receiver}?",
    ),
    (
        "{receiver} received {asset} from {sender}. Receipt identifier: {code}.",
        "Which receipt identifier records {sender} sending {asset} to {receiver}?",
    ),
    (
        "{sender} supplied {receiver} with {asset} under transfer {code}.",
        "Which transfer identifies {sender} as supplier and {receiver} as recipient of {asset}?",
    ),
    (
        "Transfer {code} moved {asset} out of the custody of {sender} and into the custody of {receiver}.",
        "What transfer reference records {asset} moving from {sender} to {receiver}?",
    ),
    (
        "{sender} handed {asset} over to {receiver}. Handover number: {code}.",
        "What handover number records {sender} handing {asset} to {receiver}?",
    ),
    (
        "The sender of {asset} was {sender}; the receiver was {receiver}; the log code was {code}.",
        "What log code identifies {sender} as the sender of {asset} to {receiver}?",
    ),
    (
        "Inbound log for {receiver}: {asset}, originating at {sender}, filed under {code}.",
        "Which log code covers {sender} shipping {asset} to {receiver}?",
    ),
    (
        "Order {code} names {sender} as provider of {asset} and {receiver} as customer.",
        "Under which order does {receiver} buy {asset} from {sender}?",
    ),
    (
        "For {asset}, the supplying unit is {sender} and the receiving unit is {receiver}. Entry: {code}.",
        "Which entry records a delivery of {asset} to {receiver} from {sender}?",
    ),
    (
        "A consignment of {asset} reached {receiver} after being dispatched by {sender}. Its tracking key is {code}.",
        "Which tracking key identifies {asset} dispatched by {sender} for {receiver}?",
    ),
    (
        "Under docket {code}, {receiver} took delivery of {asset} originating with {sender}.",
        "What is the docket number for the delivery of {asset} by {sender} to {receiver}?",
    ),
    (
        "Shipment reference {code}: {asset} went to {receiver}, having been sent by {sender}.",
        "Which reference matches {asset} received by {receiver} from {sender}?",
    ),
]

TEMPORAL = [
    (
        "The {asset} inspection in {year} was assigned to {a}. The form template has revision year {other}.",
        "Which unit was assigned to the {asset} inspection in {year}?",
    ),
    (
        "For the {year} review of {asset}, the appointed unit was {a}. The cited handbook is dated {other}.",
        "Which unit was appointed for the {year} review of {asset}?",
    ),
    (
        "{a} was responsible for the {asset} check scheduled in {year}. Reference manual edition: {other}.",
        "Which unit was responsible for the {asset} check scheduled in {year}?",
    ),
    (
        "Inspection year: {year}. Item: {asset}. Assigned unit: {a}. Worksheet edition year: {other}.",
        "Name the unit assigned to inspect {asset} in {year}.",
    ),
    (
        "The record places the {asset} audit in {year} under the responsibility of {a}. Its cited standard carries the year {other}.",
        "Who was responsible for the {asset} audit in {year}?",
    ),
    (
        "In the {year} inspection cycle, the inspection of {asset} was allocated to {a}. The report cites a catalog issued in {other}.",
        "Which unit received the inspection allocation for {asset} in {year}?",
    ),
    (
        "The assignment of {a} concerns the {year} verification of {asset}. The record uses a checklist dated {other}.",
        "Which unit is assigned to the {year} verification of {asset}?",
    ),
    (
        "The {asset} review for calendar year {year} lists {a} as its owner. Separately, the procedure document is dated {other}.",
        "Who owns the review of {asset} for calendar year {year}?",
    ),
    (
        "For {asset}, the {year} assessment names {a} as the responsible unit. Its bibliography includes a manual from {other}.",
        "Which unit is responsible for the {year} assessment of {asset}?",
    ),
    (
        "{a} holds the appointment to examine {asset} during {year}. The referenced instructions bear a {other} publication date.",
        "Which unit holds the appointment to examine {asset} during {year}?",
    ),
    (
        "The scheduled {year} evaluation of {asset} is entrusted to {a}. The form cites the {other} edition of a guide.",
        "To which unit is the {year} evaluation of {asset} entrusted?",
    ),
    (
        "Accountability for the {asset} inspection occurring in {year} rests with {a}. The accompanying reference sheet was revised in {other}.",
        "Which unit is accountable for inspecting {asset} in {year}?",
    ),
]

FAMILIES = {
    "fit": range(5),
    "selection": range(5, 7),
    "calibration": range(7, 9),
    "test": range(9, 12),
}
ASSETS = (
    "sensor kits",
    "calibration modules",
    "storage drives",
    "test fixtures",
    "power units",
    "sealed samples",
)
ACTIONS = (
    ("publish", "publishing"),
    ("archive", "archiving"),
    ("export", "exporting"),
    ("release", "releasing"),
)
PERMISSION_ASSETS = (
    "diagnostic logs",
    "release notes",
    "benchmark reports",
    "audit records",
    "configuration bundles",
    "test summaries",
)


def make_group(phenomenon, family, ordinal, role):
    group = f"contrast/{phenomenon}/{family:02}/{ordinal:03}"
    key = digest(group)
    # Fictional identifiers, with no role or label tokens embedded in the names.
    a, b = "Unit " + key[:8].upper(), "Unit " + key[8:16].upper()
    vocabulary = PERMISSION_ASSETS if phenomenon == "negation" else ASSETS
    asset = vocabulary[int(key[16:18], 16) % len(vocabulary)]
    verb, gerund = ACTIONS[int(key[18:20], 16) % len(ACTIONS)]
    values = {
        "a": a,
        "b": b,
        "asset": asset,
        "code": "REF-" + key[20:28].upper(),
        "action": verb + " the " + asset,
        "gerund": gerund + " the " + asset,
    }
    year = 2000 + int(key[28:30], 16) % 20
    years = [year, year + 1 + int(key[30:32], 16) % 5]
    contexts, questions, answers = [], [], []
    for index in range(2):
        if phenomenon == "negation":
            c, q = NEGATION[family]
            contexts.append(c[index].format(**values))
            questions.append(q[index].format(**values))
            answers.append(a)
        elif phenomenon == "relation":
            c, q = RELATION[family]
            v = dict(values, sender=[a, b][index], receiver=[b, a][index])
            contexts.append(c.format(**v))
            questions.append(q.format(**v))
            answers.append(values["code"])
        elif phenomenon == "temporal":
            c, q = TEMPORAL[family]
            v = dict(values, year=years[index], other=years[1 - index])
            contexts.append(c.format(**v))
            questions.append(q.format(**v))
            answers.append(a)
        else:
            raise ValueError("Unknown contrast phenomenon")
    rows = []
    for ci, context in enumerate(contexts):
        for qi, question in enumerate(questions):
            supported = ci == qi
            answer = answers[ci]
            rows.append(
                {
                    "id": group + f"/c{ci}q{qi}",
                    "group_id": group,
                    "document": group,
                    "role": role,
                    "suite": "contrast",
                    "phenomenon": phenomenon,
                    "template_family": f"{phenomenon}/{family:02}",
                    "question": question,
                    "context": context,
                    "answerable": supported,
                    "context_variant": ci,
                    "question_variant": qi,
                    "answers": [{"text": answer, "answer_start": context.index(answer)}]
                    if supported
                    else [],
                    "oracle": {"state": ci, "requested_state": qi, "named_answer": answer},
                    "entities": [a, b],
                    "license": "Apache-2.0",
                    "annotation": "Authored fictional counterfactual; equality oracle, no independent human adjudication",
                }
            )
    return rows


def generate(counts):
    return [
        row
        for role, families in FAMILIES.items()
        for phenomenon in ("negation", "relation", "temporal")
        for family in families
        for ordinal in range(counts[role])
        for row in make_group(phenomenon, family, ordinal, role)
    ]
