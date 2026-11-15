"""Authored counterfactual families with active/passive and polarity variation.

These are synthetic diagnostics, not independent human annotations. Families and
scenario entities are disjoint across roles. Each question and each context has
one positive and one negative partner. No metadata is passed to the model.
"""

from .support_contrasts import ASSETS, PERMISSION_ASSETS
from .support_data import digest

# A family is an indivisible split unit. Negation has two explicit states;
# relations swap sender/receiver; temporal cases swap event/reference year.
NEGATION = {
    "fit": [
        (
            ("{a} received clearance to {action}.", "{a} was denied clearance to {action}."),
            (
                "Which unit has authorization to {action}?",
                "Which unit was refused authorization to {action}?",
            ),
        ),
        (
            (
                "The decision expressly authorizes {a} to {action}.",
                "The decision expressly prohibits {a} from {gerund}.",
            ),
            ("Which unit is permitted to {action}?", "Which unit is prohibited from {gerund}?"),
        ),
        (
            ("{a} has been approved for {gerund}.", "{a} has not been approved for {gerund}."),
            (
                "Name the unit with approval to {action}.",
                "Name the unit without approval to {action}.",
            ),
        ),
        (
            (
                "{a} is exempt from the ban on {gerund}.",
                "{a} remains subject to the ban on {gerund}.",
            ),
            (
                "Which unit may {action} despite the general ban?",
                "Which unit is still forbidden to {action}?",
            ),
        ),
    ],
    "selection": [
        (
            (
                "Authorization for {a} to {action} is in force.",
                "Authorization for {a} to {action} is not in force.",
            ),
            (
                "Which unit currently has permission to {action}?",
                "Which unit currently lacks permission to {action}?",
            ),
        ),
        (
            (
                "{a} meets the authorization requirement for {gerund}.",
                "{a} fails the authorization requirement for {gerund}.",
            ),
            (
                "Which unit satisfies the permission requirement to {action}?",
                "Which unit does not satisfy the permission requirement to {action}?",
            ),
        ),
        (
            (
                "The committee accepted {a}'s application to {action}.",
                "The committee declined {a}'s application to {action}.",
            ),
            (
                "Whose application to {action} was successful?",
                "Whose application to {action} was unsuccessful?",
            ),
        ),
    ],
    "calibration": [
        (
            ("{a} holds a valid permit to {action}.", "{a} holds no valid permit to {action}."),
            (
                "Which unit has a valid permit for {gerund}?",
                "Which unit has no valid permit for {gerund}?",
            ),
        ),
        (
            (
                "The signed instruction permits {a} to {action}.",
                "The signed instruction does not permit {a} to {action}.",
            ),
            (
                "Which unit is allowed to {action} by the signed instruction?",
                "Which unit is not allowed to {action} by the signed instruction?",
            ),
        ),
        (
            ("{a} secured approval to {action}.", "{a} failed to secure approval to {action}."),
            (
                "Which unit obtained approval for {gerund}?",
                "Which unit did not obtain approval for {gerund}?",
            ),
        ),
    ],
    "test": [
        (
            (
                "The approval board gave {a} the go-ahead to {action}.",
                "The approval board withheld the go-ahead for {a} to {action}.",
            ),
            (
                "Which unit received permission to {action} from the approval board?",
                "Which unit did not receive permission to {action} from the approval board?",
            ),
        ),
        (
            (
                "{a} is on the list of units allowed to {action}.",
                "{a} is not on the list of units allowed to {action}.",
            ),
            (
                "Which unit appears on the approved list for {gerund}?",
                "Which unit is absent from the approved list for {gerund}?",
            ),
        ),
        (
            (
                "The restriction on {a} {gerund} has been lifted.",
                "The restriction on {a} {gerund} has not been lifted.",
            ),
            (
                "Which unit is no longer restricted from {gerund}?",
                "Which unit is still restricted from {gerund}?",
            ),
        ),
        (
            (
                "{a} has the necessary authorization for {gerund}.",
                "{a} is missing the necessary authorization for {gerund}.",
            ),
            (
                "Which unit possesses the authorization needed to {action}?",
                "Which unit lacks the authorization needed to {action}?",
            ),
        ),
        (
            (
                "The policy exempts {a} from the prohibition against {gerund}.",
                "The policy does not exempt {a} from the prohibition against {gerund}.",
            ),
            (
                "Which unit is excepted from the prohibition against {gerund}?",
                "Which unit is not excepted from the prohibition against {gerund}?",
            ),
        ),
    ],
}

RELATION = {
    "fit": [
        (
            "{asset} were sent by {sender} to {receiver}. Record: {code}.",
            "Which record describes {receiver} receiving {asset} from {sender}?",
        ),
        (
            "{receiver} obtained {asset} from {sender}, recorded as {code}.",
            "What record identifies the transfer of {asset} by {sender} to {receiver}?",
        ),
        (
            "Dispatch {code}: {sender} is the origin, {receiver} the destination, and {asset} the cargo.",
            "Which dispatch routes {asset} to {receiver} from {sender}?",
        ),
        (
            "{asset} were delivered to {receiver} by {sender}; reference {code}.",
            "What reference names {sender} as the source of {asset} for {receiver}?",
        ),
        (
            "The transaction {code} lists {sender} as lender and {receiver} as borrower of {asset}.",
            "Which transaction records {receiver} borrowing {asset} from {sender}?",
        ),
        (
            "{sender} sold {asset} to {receiver} in sale {code}.",
            "Which sale records {receiver} buying {asset} from {sender}?",
        ),
    ],
    "selection": [
        (
            "{receiver} acquired {asset} supplied by {sender}. Acquisition code: {code}.",
            "Which acquisition code names {sender} as supplier to {receiver} of {asset}?",
        ),
        (
            "In transfer {code}, {asset} departed {sender} and arrived at {receiver}.",
            "Which transfer took {asset} to {receiver} from {sender}?",
        ),
        (
            "{sender} loaned {asset} to {receiver}, with agreement {code}.",
            "Which agreement makes {receiver} the borrower of {asset} from {sender}?",
        ),
    ],
    "calibration": [
        (
            "Receiving party: {receiver}. Sending party: {sender}. Contents: {asset}. Voucher: {code}.",
            "Which voucher describes {sender} shipping {asset} to {receiver}?",
        ),
        (
            "{asset} left the inventory of {sender} for that of {receiver} under entry {code}.",
            "Which entry records {receiver} taking {asset} from {sender}'s inventory?",
        ),
        (
            "{receiver} purchased {asset} from {sender}; purchase order {code}.",
            "Which purchase order identifies {sender} selling {asset} to {receiver}?",
        ),
    ],
    "test": [
        (
            "The delivery documented by {code} originated with {sender}; {asset} were accepted by {receiver}.",
            "Which document records {receiver} accepting {asset} originating with {sender}?",
        ),
        (
            "{asset} passed from {sender}'s stock into {receiver}'s stock, ledger reference {code}.",
            "Which ledger reference concerns {asset} acquired by {receiver} out of {sender}'s stock?",
        ),
        (
            "{receiver} was the addressee for {asset} mailed by {sender}, mailing receipt {code}.",
            "Which mailing receipt lists {sender} mailing {asset} to {receiver}?",
        ),
        (
            "Rental {code} assigns {asset} owned by {sender} to the renter {receiver}.",
            "Which rental lets {receiver} rent {asset} from their owner {sender}?",
        ),
        (
            "The outgoing unit for {asset} is {sender}; the incoming unit is {receiver}; movement identifier {code}.",
            "Which movement identifier describes {asset} incoming to {receiver} from {sender}?",
        ),
    ],
}

TEMPORAL = {
    "fit": [
        (
            "The {asset} examination occurred in {year}, conducted by {a}. The checklist was published in {other}.",
            "Which unit conducted the {asset} examination in {year}?",
        ),
        (
            "{a} led the {year} evaluation of {asset}. The record also cites the {other} handbook edition.",
            "Who led the evaluation of {asset} during {year}?",
        ),
        (
            "Year of the {asset} assessment: {year}; responsible team: {a}. The assessment cites the {other} manual.",
            "Name the team responsible for the {year} assessment of {asset}.",
        ),
    ],
    "selection": [
        (
            "{a} carried out the {asset} review during {year}. The review protocol dates from {other}.",
            "Which unit carried out the review of {asset} during {year}?",
        ),
        (
            "The log assigns the {year} examination of {asset} to {a}, citing guidelines issued in {other}.",
            "Who was assigned to examine {asset} in {year}?",
        ),
        (
            "The inspection of {asset} took place in {year} under {a}. The record separately cites instructions printed in {other}.",
            "Which unit oversaw the inspection of {asset} that took place in {year}?",
        ),
    ],
    "calibration": [
        (
            "A review of {asset} was performed by {a} in {year}; the cited specification was released in {other}.",
            "Which unit performed the {asset} review in {year}?",
        ),
        (
            "{a} managed the inspection of {asset} for {year}. The reference guide's edition is {other}.",
            "Who managed the inspection of {asset} for {year}?",
        ),
        (
            "The {asset} testing cycle in {year} was coordinated by {a}; the form refers to a {other} standard.",
            "Which unit coordinated the {asset} testing cycle in {year}?",
        ),
    ],
    "test": [
        (
            "The evaluation of {asset} is dated {year} and credits {a} as the lead unit. The cited operating guide bears the date {other}.",
            "Which lead unit is credited for the evaluation of {asset} dated {year}?",
        ),
        (
            "{a} took charge of the {asset} appraisal in {year}. The file also lists the {other} edition of the instructions.",
            "Who took charge of the {asset} appraisal in {year}?",
        ),
        (
            "For the {asset} check that occurred in {year}, the designated unit was {a}. The worksheet's copyright year is {other}.",
            "Which unit was designated for the {asset} check that occurred in {year}?",
        ),
        (
            "{a} supervised the {year} audit of {asset}. The attached handbook was last updated in {other}.",
            "Which unit supervised the audit of {asset} in {year}?",
        ),
        (
            "The file confirms that {a} handled the examination of {asset} in {year}. Its appendix reproduces a procedure from {other}.",
            "Who handled the examination of {asset} in {year}?",
        ),
    ],
}

FAMILIES = {"negation": NEGATION, "relation": RELATION, "temporal": TEMPORAL}
NAMES = (
    "Amber",
    "Cedar",
    "Flint",
    "Maple",
    "Pine",
    "Coral",
    "Silver",
    "Willow",
    "Jade",
    "Slate",
    "Birch",
    "Copper",
)


def make_group(phenomenon, role, family, ordinal):
    group = f"v3/{phenomenon}/{role}/{family:02}/{ordinal:03}"
    key = digest(group)
    first = int(key[:2], 16) % len(NAMES)
    second = (first + 1 + int(key[2:4], 16) % (len(NAMES) - 1)) % len(NAMES)
    # Readable, distinct first words plus scenario-unique suffixes, unlike only hex IDs.
    a, b = f"{NAMES[first]} unit {key[4:12]}", f"{NAMES[second]} unit {key[12:20]}"
    assets = PERMISSION_ASSETS if phenomenon == "negation" else ASSETS
    asset = assets[int(key[12:14], 16) % len(assets)]
    verb, gerund = (("publish", "publishing"), ("export", "exporting"), ("archive", "archiving"))[
        int(key[14:16], 16) % 3
    ]
    year = 1995 + int(key[16:18], 16) % 25
    years = [year, year + 1 + int(key[18:20], 16) % 7]
    values = dict(
        a=a,
        b=b,
        asset=asset,
        code="MOVE-" + key[20:28],
        action=verb + " the " + asset,
        gerund=gerund + " the " + asset,
    )
    contexts, questions = [], []
    c, q = FAMILIES[phenomenon][role][family]
    for state in range(2):
        v = dict(
            values,
            sender=[a, b][state],
            receiver=[b, a][state],
            year=years[state],
            other=years[1 - state],
        )
        context = (c[state] if phenomenon == "negation" else c).format(**v)
        context = context[0].upper() + context[1:]
        question = (q[state] if phenomenon == "negation" else q).format(**v)
        # A label-independent distractor, before/after main fact, same in both states.
        distractor = (
            "This record concerns a single transaction."
            if phenomenon == "relation"
            else "This record is a complete description of this assignment."
        )
        context = (
            (distractor + " " + context) if int(key[28], 16) % 2 else (context + " " + distractor)
        )
        contexts.append(context)
        questions.append(question)
    answer = values["code"] if phenomenon == "relation" else a
    return [
        dict(
            id=group + f"/c{ci}q{qi}",
            group_id=group,
            document=group,
            role=role,
            suite="contrast",
            phenomenon=phenomenon,
            template_family=f"v3/{phenomenon}/{role}/{family:02}",
            question=question,
            context=context,
            answerable=ci == qi,
            context_variant=ci,
            question_variant=qi,
            answers=[{"text": answer, "answer_start": context.index(answer)}] if ci == qi else [],
            oracle={"state": ci, "requested_state": qi, "named_answer": answer},
            entities=[a, b],
            license="Apache-2.0",
            annotation="Authored counterfactual; no independent adjudication",
        )
        for ci, context in enumerate(contexts)
        for qi, question in enumerate(questions)
    ]


def generate(counts):
    return [
        r
        for phenomenon, roles in FAMILIES.items()
        for role, families in roles.items()
        for family in range(len(families))
        for ordinal in range(counts[role])
        for r in make_group(phenomenon, role, family, ordinal)
    ]
