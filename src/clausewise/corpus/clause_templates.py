"""Clause templates: the sentence pools the corpus generator draws from.

Each template is a bag of sentence slots. A clause is one alternative from every required slot plus a
seeded subset of the optional slots, so a label with three slots of three alternatives already yields
27 distinct bodies before placeholder values and party roles vary. Risk-bearing labels carry several
variants: the risky wording, the ordinary wording, and a hard negative that reuses the risky vocabulary
("unlimited", "automatically renew", "terminate for convenience") in a sentence that carries no risk.
The hard negatives exist so that the risk benchmark punishes keyword matching.

Placeholders: ``{A}``/``{B}`` are the two party roles, ``{U}``/``{V}`` the party holding a one-sided
right and its counterpart, ``{ASSIGNOR}``/``{ASSIGNEE}`` the parties to an IP transfer, ``{ref:label}``
a cross reference rendered with the contract's numbering, and lowercase keys are generated values.
No template contains a literal duration, amount, date or percentage: every such value comes through a
placeholder so that the entity ground truth is complete.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClauseTemplate:
    """A pool of sentences for one clause label and variant."""

    label: str
    variant: str
    headings: tuple[str, ...]
    slots: tuple[tuple[str, ...], ...]
    optional: tuple[tuple[str, ...], ...] = ()
    risk: str | None = None
    contract_types: tuple[str, ...] | None = None


def _t(
    label: str,
    variant: str,
    headings: tuple[str, ...],
    slots: tuple[tuple[str, ...], ...],
    optional: tuple[tuple[str, ...], ...] = (),
    risk: str | None = None,
    contract_types: tuple[str, ...] | None = None,
) -> ClauseTemplate:
    return ClauseTemplate(label, variant, headings, slots, optional, risk, contract_types)


H_DEFINITIONS = ("Definitions", "Definitions and Interpretation", "Interpretation")
H_CONFIDENTIALITY = ("Confidentiality", "Confidential Information", "Non-Disclosure", "Confidentiality Obligations")
H_TERM = ("Term", "Duration", "Term of Agreement", "Term and Renewal")
H_TERMINATION = ("Termination", "Termination Rights", "Ending this Agreement", "Termination of Agreement")
H_LIABILITY = ("Limitation of Liability", "Liability", "Limitations of Liability", "Liability Cap")
H_IP = ("Intellectual Property", "Intellectual Property Rights", "Ownership", "Proprietary Rights")
H_NON_COMPETE = ("Non-Competition", "Non-Compete", "Restrictive Covenants", "Restrictions on Competition")
H_GOVERNING_LAW = ("Governing Law", "Applicable Law", "Governing Law and Jurisdiction", "Law")

TEMPLATES: list[ClauseTemplate] = [
    _t(
        "definitions",
        "base",
        H_DEFINITIONS,
        (
            (
                "In this Agreement, unless the context otherwise requires, capitalised terms have the meanings set out "
                "in this clause.",
                "The following definitions apply in this Agreement.",
                "Capitalised terms used in this Agreement have the meanings given below.",
            ),
            (
                '"Affiliate" means any entity that controls, is controlled by or is under common control with a party.',
                '"Business Day" means a day other than a Saturday, Sunday or public holiday in {city}.',
                '"Applicable Law" means all laws, statutes, regulations and codes applicable to a party or to the '
                "subject matter of this Agreement.",
            ),
            (
                '"Confidential Information" means all information disclosed by one party to the other that is marked '
                "confidential or would reasonably be understood to be confidential, as further described in "
                "{ref:confidentiality}.",
                '"Effective Date" means the date first written above.',
                '"Term" has the meaning given in {ref:term}.',
            ),
        ),
        (
            (
                "Clause headings are for convenience only and do not affect interpretation.",
                "Words in the singular include the plural and vice versa, and references to a person include a "
                "company, partnership or other legal entity.",
            ),
        ),
    ),
    _t(
        "confidentiality",
        "base",
        H_CONFIDENTIALITY,
        (
            (
                "{B} shall keep the Confidential Information of {A} strictly confidential and shall not disclose it to "
                "any third party without the prior written consent of {A}.",
                "Each party shall hold the other party's Confidential Information in confidence and shall use it "
                "solely "
                "for the purposes of this Agreement.",
                "{B} agrees to protect the Confidential Information using at least the same degree of care it uses to "
                "protect its own confidential information, and in no case less than reasonable care.",
            ),
            (
                "{B} may disclose Confidential Information only to its employees, officers and professional advisers "
                "who need to know it for the purposes of this Agreement and who are bound by written obligations of "
                "confidentiality no less protective than those in this Agreement.",
                "Confidential Information may be disclosed to the extent required by law, regulation or court order, "
                "provided that the receiving party gives the disclosing party prompt written notice and cooperates in "
                "seeking a protective order.",
                "Neither party shall copy or reproduce the other party's Confidential Information except as strictly "
                "necessary for the purposes of this Agreement.",
            ),
            (
                "The obligations in this clause shall survive termination of this Agreement for a period of "
                "{conf_duration}.",
                "This clause shall continue in force for {conf_duration} after the date of termination or expiry of "
                "this Agreement.",
                "The confidentiality obligations set out in this clause shall remain in effect for {conf_duration} "
                "following the date of disclosure.",
            ),
        ),
        (
            (
                "Nothing in this Agreement shall prevent a party from making a disclosure required by applicable law "
                "or the rules of any stock exchange.",
                "Each party shall notify the other promptly on becoming aware of any unauthorised use or disclosure "
                "of Confidential Information.",
            ),
        ),
    ),
    _t(
        "exclusions",
        "base",
        ("Exclusions", "Exceptions", "Excluded Information"),
        (
            (
                "Confidential Information does not include information that (a) is or becomes publicly available "
                "other than through a breach of this Agreement; (b) was lawfully in the possession of {B} before "
                "disclosure; (c) is independently developed by {B} without use of the Confidential Information; or "
                "(d) is received from a third party free of any obligation of confidence.",
                "The obligations in {ref:confidentiality} do not apply to information which {B} can show by written "
                "records was already known to it, is in the public domain through no fault of {B}, or was developed "
                "independently without reference to the Confidential Information.",
            ),
        ),
        (
            (
                "The burden of proving that an exclusion applies shall rest with {B}.",
                "A combination of features shall not be deemed public merely because individual features are public.",
            ),
        ),
        contract_types=("nda",),
    ),
    _t(
        "term",
        "fixed",
        H_TERM,
        (
            (
                "This Agreement shall commence on the Effective Date and shall continue for {term} unless terminated "
                "earlier in accordance with {ref:termination}.",
                "This Agreement takes effect on the Effective Date and remains in force for an initial term of {term}.",
                "The term of this Agreement is {term} from the Effective Date.",
            ),
            (
                "On expiry of the initial term this Agreement shall terminate unless the parties agree in writing to "
                "extend it.",
                "Any extension of the term requires a written agreement signed by both parties.",
                "This Agreement shall expire at the end of the term without the need for notice by either party.",
            ),
        ),
    ),
    _t(
        "term",
        "auto_renew",
        H_TERM,
        (
            (
                "This Agreement shall commence on the Effective Date and shall continue for an initial term of {term}.",
                "The initial term of this Agreement is {term} starting on the Effective Date.",
                "This Agreement takes effect on the Effective Date and remains in force for {term}.",
            ),
            (
                "Thereafter this Agreement shall automatically renew for successive periods of {renewal} unless "
                "either party gives written notice of non-renewal at least {nonrenew_notice} before the end of the "
                "then-current term.",
                "Upon expiry of the initial term, this Agreement shall renew automatically for further terms of "
                "{renewal} each, unless terminated by either party by written notice given no later than "
                "{nonrenew_notice} prior to the expiry of the current term.",
                "This Agreement shall be extended automatically for consecutive renewal terms of {renewal} unless "
                "{B} notifies {A} in writing of its intention not to renew at least {nonrenew_notice} before the "
                "renewal date.",
            ),
        ),
        risk="auto_renewal",
    ),
    _t(
        "term",
        "hard_negative",
        H_TERM,
        (
            (
                "This Agreement shall commence on the Effective Date and shall continue for {term}.",
                "The initial term of this Agreement is {term} from the Effective Date.",
            ),
            (
                "This Agreement shall not renew automatically; any renewal shall require a new written agreement "
                "signed by both parties.",
                "Renewal of this Agreement is at the discretion of both parties and must be agreed in writing at "
                "least {nonrenew_notice} before expiry; there is no automatic renewal.",
                "The parties may renew this Agreement for a further {renewal} by mutual written agreement, failing "
                "which it shall expire at the end of the term.",
            ),
        ),
    ),
    _t(
        "termination",
        "mutual",
        H_TERMINATION,
        (
            (
                "Either party may terminate this Agreement for convenience by giving the other party not less than "
                "{notice} prior written notice.",
                "This Agreement may be terminated by either party without cause on {notice} written notice to the "
                "other party.",
            ),
            (
                "Either party may terminate this Agreement with immediate effect by written notice if the other party "
                "commits a material breach which, if capable of remedy, is not remedied within {cure} of receipt of "
                "written notice requiring remedy.",
                "Either party may terminate this Agreement immediately if the other party becomes insolvent, enters "
                "into liquidation or has a receiver appointed over any of its assets.",
            ),
            (
                "Upon termination, {B} shall pay all amounts accrued up to the date of termination and each party "
                "shall return or destroy the other party's Confidential Information.",
                "Termination shall not affect any rights or remedies that have accrued before termination.",
                "Clauses which by their nature are intended to survive termination shall survive.",
            ),
        ),
    ),
    _t(
        "termination",
        "for_cause_only",
        H_TERMINATION,
        (
            (
                "This Agreement may be terminated only in accordance with this clause and neither party may "
                "terminate for convenience.",
                "Neither party may terminate this Agreement before the end of the term except as set out in this "
                "clause.",
            ),
            (
                "Either party may terminate this Agreement with immediate effect by written notice if the other party "
                "commits a material breach which, if capable of remedy, is not remedied within {cure} of receipt of "
                "written notice requiring remedy.",
                "Either party may terminate this Agreement immediately if the other party becomes insolvent or ceases "
                "to carry on business.",
            ),
            (
                "Termination shall not affect any rights or remedies that have accrued before termination.",
                "Clauses which by their nature are intended to survive termination shall survive.",
            ),
        ),
    ),
    _t(
        "termination",
        "unilateral",
        H_TERMINATION,
        (
            (
                "{U} may terminate this Agreement at any time for any reason by giving {V} {notice} written notice. "
                "{V} may terminate this Agreement only for material breach by {U} that remains unremedied after "
                "{cure}.",
                "{U} may terminate this Agreement for convenience on {notice} written notice to {V}. {V} shall have "
                "no right to terminate this Agreement for convenience.",
                "{U} reserves the right to terminate this Agreement, in whole or in part, at its sole discretion upon "
                "{notice} notice to {V}. {V} may terminate only for an uncured material breach by {U} after {cure} "
                "written notice.",
            ),
            (
                "Upon termination, {B} shall pay all amounts accrued up to the date of termination.",
                "Termination shall not affect any rights or remedies that have accrued before termination.",
                "Clauses which by their nature are intended to survive termination shall survive.",
            ),
        ),
        risk="unilateral_termination",
    ),
    _t(
        "termination",
        "hard_negative",
        H_TERMINATION,
        (
            (
                "Either party may terminate for convenience on {notice} written notice; no party shall have any "
                "greater or lesser right to terminate than the other.",
                "Termination for convenience is available to both parties on equal terms upon {notice} written "
                "notice, and neither {A} nor {B} may terminate at its sole discretion without such notice.",
            ),
            (
                "Either party may also terminate this Agreement immediately if the other party commits a material "
                "breach that is not remedied within {cure} of written notice.",
                "In addition, either party may terminate immediately on the insolvency of the other party.",
            ),
            (
                "Termination shall not affect any rights or remedies that have accrued before termination.",
                "Clauses which by their nature are intended to survive termination shall survive.",
            ),
        ),
    ),
    _t(
        "liability",
        "capped",
        H_LIABILITY,
        (
            (
                "Except for liability arising from fraud, death or personal injury caused by negligence, or breach of "
                "{ref:confidentiality}, the total aggregate liability of each party under or in connection with this "
                "Agreement shall not exceed {cap}.",
                "Each party's total liability arising out of or in connection with this Agreement, whether in "
                "contract, tort (including negligence) or otherwise, shall be limited to {cap}.",
                "In no event shall either party's aggregate liability under this Agreement exceed {cap}.",
            ),
            (
                "Neither party shall be liable for any indirect, consequential or special loss, loss of profit, loss "
                "of business or loss of data.",
                "Neither party shall be liable to the other for any loss of profits, loss of revenue or indirect or "
                "consequential loss, however arising.",
            ),
        ),
    ),
    _t(
        "liability",
        "unlimited",
        H_LIABILITY,
        (
            (
                "{B} shall be liable for all losses, damages, costs and expenses suffered by {A} arising out of or in "
                "connection with this Agreement, without any limitation as to amount.",
                "The liability of {B} under this Agreement shall be unlimited.",
                "Nothing in this Agreement shall limit or exclude the liability of {B} to {A}, which shall be "
                "unlimited in amount and shall include indirect and consequential losses.",
            ),
        ),
        (
            (
                "The liability of {A} to {B} under this Agreement shall be limited to {cap}.",
                "{B} acknowledges that this allocation of risk is reflected in the commercial terms of this Agreement.",
            ),
        ),
        risk="unlimited_liability",
    ),
    _t(
        "liability",
        "hard_negative",
        H_LIABILITY,
        (
            (
                "The liability of each party shall not be unlimited and shall in all cases be capped at {cap}.",
                "Nothing in this clause shall be read as making either party's liability unlimited; the aggregate "
                "liability of each party is limited to {cap}.",
                "Neither party's liability shall be unlimited: each party's aggregate liability is limited to {cap}, "
                "except in the case of fraud.",
            ),
            (
                "Neither party shall be liable for any indirect or consequential loss or for loss of profit.",
                "The exclusions in this clause do not apply to liability which cannot be limited by law.",
            ),
        ),
    ),
    _t(
        "indemnification",
        "base",
        ("Indemnification", "Indemnity", "Indemnities"),
        (
            (
                "{A} shall indemnify, defend and hold harmless {B} from and against all claims, losses, damages and "
                "expenses arising from any third-party claim that the services or deliverables infringe the "
                "intellectual property rights of a third party.",
                "Each party (the indemnifying party) shall indemnify the other party against all liabilities, costs "
                "and expenses arising from the indemnifying party's breach of this Agreement or negligence.",
                "{B} shall indemnify {A} against all losses arising from {B}'s use of the services in breach of this "
                "Agreement or of applicable law.",
            ),
            (
                "The indemnified party shall promptly notify the indemnifying party of any claim and shall allow the "
                "indemnifying party to control the defence and settlement of the claim.",
                "The indemnity in this clause shall not apply to the extent the claim results from the indemnified "
                "party's own negligence or breach.",
            ),
        ),
    ),
    _t(
        "ip",
        "retain",
        H_IP,
        (
            (
                "Each party retains all rights in its pre-existing intellectual property. {A} grants {B} a "
                "non-exclusive, non-transferable licence to use the deliverables for its internal business purposes "
                "during the term.",
                "All intellectual property rights in the services and the deliverables shall remain vested in {A}. "
                "Nothing in this Agreement transfers ownership of any intellectual property from one party to the "
                "other.",
                "{A} retains ownership of all intellectual property in its software, tools and methodologies, and "
                "grants {B} a licence to use them solely as part of the services.",
            ),
        ),
        (
            (
                "{B} shall not remove or alter any proprietary notices on the deliverables.",
                "Any feedback provided by {B} may be used by {A} without restriction.",
            ),
        ),
    ),
    _t(
        "ip",
        "deliverables",
        H_IP,
        (
            (
                "Upon payment in full, {ASSIGNOR} assigns to {ASSIGNEE} all intellectual property rights in the "
                "deliverables created specifically for {ASSIGNEE} under this Agreement, excluding {ASSIGNOR}'s "
                "pre-existing materials and tools, in which {ASSIGNOR} retains all rights and grants {ASSIGNEE} a "
                "perpetual licence.",
                "{ASSIGNEE} shall own the work product created exclusively for it under a statement of work. "
                "{ASSIGNOR}'s background intellectual property remains the property of {ASSIGNOR} and is licensed to "
                "{ASSIGNEE} to the extent incorporated in the work product.",
            ),
        ),
        (
            (
                "Each party shall execute such documents as are reasonably necessary to give effect to this clause.",
                "Inventions made by {ASSIGNOR} in the course of performing this Agreement and relating to the "
                "deliverables shall be disclosed to {ASSIGNEE} promptly.",
            ),
        ),
    ),
    _t(
        "ip",
        "broad",
        H_IP,
        (
            (
                "{ASSIGNOR} hereby irrevocably assigns to {ASSIGNEE}, with full title guarantee, all present and "
                "future intellectual property rights in any work, invention, improvement or material created by "
                "{ASSIGNOR}, whether or not created in the course of this Agreement, including {ASSIGNOR}'s "
                "pre-existing intellectual property used in or related to the deliverables.",
                "All intellectual property conceived, developed or reduced to practice by {ASSIGNOR} during the term "
                "of this Agreement, whether or not related to the services and including background intellectual "
                "property, shall vest in and is hereby assigned to {ASSIGNEE} absolutely.",
                "{ASSIGNOR} assigns to {ASSIGNEE} all right, title and interest in and to all intellectual property "
                "created by {ASSIGNOR}, including pre-existing works and any improvements to them, and waives all "
                "moral rights in such works.",
            ),
        ),
        (
            (
                "{ASSIGNOR} shall execute all documents and do all things necessary to perfect the assignment in this "
                "clause, at {ASSIGNEE}'s request and without further consideration.",
                "This assignment shall survive termination of this Agreement.",
            ),
        ),
        risk="ip_assignment",
    ),
    _t(
        "ip",
        "hard_negative",
        H_IP,
        (
            (
                "Nothing in this Agreement shall operate to assign any pre-existing intellectual property of either "
                "party; each party retains ownership of its background intellectual property and grants only the "
                "licences expressly set out in this clause.",
                "No assignment of intellectual property is made under this Agreement. {ASSIGNOR} grants {ASSIGNEE} a "
                "limited licence to the deliverables, and the parties acknowledge that any assignment of background "
                "intellectual property would require a separate written agreement.",
            ),
        ),
        (
            (
                "{ASSIGNEE} shall not reverse engineer or decompile any software provided under this Agreement.",
                "Each party shall promptly notify the other of any suspected infringement of the licensed rights.",
            ),
        ),
    ),
    _t(
        "warranties",
        "base",
        ("Warranties", "Representations and Warranties", "Warranty"),
        (
            (
                "{A} warrants that the services will be performed with reasonable skill and care by suitably qualified "
                "personnel and in accordance with good industry practice.",
                "Each party warrants that it has full power and authority to enter into and perform this Agreement "
                "and that doing so will not breach any other agreement to which it is a party.",
            ),
            (
                "Except as expressly set out in this Agreement, all warranties, conditions and terms, whether express "
                "or implied by statute or common law, are excluded to the fullest extent permitted by law.",
                "{B}'s sole remedy for breach of the warranty in this clause shall be the re-performance of the "
                "defective services at no additional charge.",
            ),
        ),
    ),
    _t(
        "non_compete",
        "post_term",
        H_NON_COMPETE,
        (
            (
                "{B} shall not, during the term of this Agreement and for a period of {noncompete} after its "
                "termination, directly or indirectly engage in, be employed by or hold any interest in any business "
                "that competes with the business of {A} within {territory}.",
                "For {noncompete} following termination of this Agreement for any reason, {B} shall not carry on or "
                "be engaged in any business which competes with {A} in {territory}, without the prior written "
                "consent of {A}.",
                "{B} covenants that it shall not, for {noncompete} after the end of this Agreement, work for or "
                "provide services to a competitor of {A} in {territory}.",
            ),
        ),
        (
            (
                "{B} acknowledges that the restrictions in this clause are reasonable and necessary to protect the "
                "legitimate business interests of {A}.",
                "Each restriction in this clause is a separate undertaking and, if held unenforceable, shall be "
                "severed without affecting the remaining restrictions.",
            ),
        ),
        risk="non_compete",
    ),
    _t(
        "non_compete",
        "during_term",
        H_NON_COMPETE,
        (
            (
                "During the term of this Agreement, {B} shall not provide services to any direct competitor of {A} "
                "without {A}'s prior written consent. This restriction ends on termination of this Agreement and "
                "does not restrict {B}'s activities after termination.",
                "{B} agrees not to engage in any competing business while this Agreement is in force. Nothing in "
                "this Agreement restricts {B} from competing with {A} after the Agreement ends.",
            ),
        ),
        (
            (
                "{B} shall disclose to {A} any actual or potential conflict of interest arising during the term.",
                "For the purposes of this clause a competitor means any business offering products or services "
                "substantially similar to those of {A}.",
            ),
        ),
    ),
    _t(
        "non_compete",
        "hard_negative",
        H_NON_COMPETE,
        (
            (
                "Nothing in this Agreement shall prevent {B} from engaging in any business or employment after "
                "termination, including with competitors of {A}; the parties agree that no non-compete restriction "
                "applies.",
                "The parties acknowledge that this Agreement contains no covenant not to compete, and {B} remains "
                "free to compete with {A} both during and after the term.",
            ),
        ),
        (
            (
                "{B} shall nevertheless continue to comply with {ref:confidentiality} after termination.",
                "This clause does not limit any obligation of confidentiality under this Agreement.",
            ),
        ),
    ),
    _t(
        "non_solicit",
        "base",
        ("Non-Solicitation", "Non-Solicit", "Solicitation of Employees"),
        (
            (
                "During the term and for {nonsolicit} after termination, neither party shall directly or indirectly "
                "solicit for employment any employee of the other party with whom it had material contact under this "
                "Agreement, save through general advertising not targeted at such employees.",
                "{B} shall not, for {nonsolicit} after termination of this Agreement, solicit or entice away from {A} "
                "any person who was an employee or consultant of {A} during the final year of the term.",
            ),
        ),
        (
            (
                "This clause does not prevent the hiring of any person who responds to a public job advertisement.",
                "The parties agree that this restriction is no wider than is reasonably necessary.",
            ),
        ),
    ),
    _t(
        "governing_law",
        "base",
        H_GOVERNING_LAW,
        (
            (
                "This Agreement and any dispute or claim arising out of or in connection with it shall be governed by "
                "and construed in accordance with {law}.",
                "This Agreement shall be governed by {law}, without regard to its conflict of laws principles.",
                "The validity, construction and performance of this Agreement shall be governed by {law}.",
            ),
        ),
        (
            (
                "Each party irrevocably submits to the exclusive jurisdiction of {courts} in relation to any dispute "
                "arising under this Agreement.",
                "The parties submit to the non-exclusive jurisdiction of {courts}.",
            ),
        ),
    ),
    _t(
        "dispute_resolution",
        "base",
        ("Dispute Resolution", "Disputes", "Arbitration", "Resolution of Disputes"),
        (
            (
                "Any dispute arising out of or in connection with this Agreement, including any question regarding "
                "its existence, validity or termination, shall be referred to and finally resolved by arbitration "
                "administered by {arb_institution} in accordance with its rules. The seat of arbitration shall be "
                "{seat}, the tribunal shall consist of a sole arbitrator and the language of the arbitration shall "
                "be English.",
                "The parties shall attempt in good faith to resolve any dispute through negotiation between senior "
                "executives within {dispute_days} of written notice of the dispute. If the dispute is not resolved, "
                "either party may refer it to {courts}.",
                "Any dispute shall first be referred to mediation; if it is not settled within {dispute_days}, the "
                "dispute shall be submitted to the exclusive jurisdiction of {courts}.",
            ),
        ),
        (
            (
                "Nothing in this clause prevents a party from seeking urgent injunctive relief from any court of "
                "competent jurisdiction.",
                "The parties shall continue to perform their obligations under this Agreement while a dispute is "
                "being resolved.",
            ),
        ),
    ),
    _t(
        "assignment",
        "base",
        ("Assignment", "Assignment and Subcontracting", "Transfer of Rights"),
        (
            (
                "Neither party may assign, transfer or subcontract any of its rights or obligations under this "
                "Agreement without the prior written consent of the other party, such consent not to be unreasonably "
                "withheld.",
                "{B} shall not assign this Agreement without the prior written consent of {A}. {A} may assign this "
                "Agreement to any Affiliate or to a successor in interest upon written notice to {B}.",
            ),
        ),
        (
            (
                "Any purported assignment in breach of this clause shall be void.",
                "This Agreement shall bind and benefit the parties and their permitted successors and assigns.",
            ),
        ),
        contract_types=("nda", "msa", "saas", "employment"),
    ),
    _t(
        "assignment",
        "lease",
        ("Assignment and Subletting", "Subletting", "Assignment"),
        (
            (
                "{B} shall not assign this Lease or sublet the whole or any part of the Premises without the prior "
                "written consent of {A}, which may be withheld at {A}'s reasonable discretion.",
                "{B} may not transfer its interest in the Premises or permit any other person to occupy the Premises "
                "without the prior written consent of {A}.",
            ),
        ),
        (
            (
                "Any assignment or subletting in breach of this clause shall entitle {A} to terminate this Lease.",
                "{A} may assign its interest in the Premises to a purchaser of the building on written notice to {B}.",
            ),
        ),
        contract_types=("lease",),
    ),
    _t(
        "notices",
        "base",
        ("Notices", "Notice", "Communications"),
        (
            (
                "Any notice under this Agreement shall be in writing and delivered by hand, by registered post or by "
                "email to the address of the receiving party set out above, or such other address as a party may "
                "notify in writing.",
                "All notices shall be in writing and sent to the registered address or notified email address of the "
                "receiving party.",
            ),
            (
                "Notices shall be deemed received: if delivered by hand, on delivery; if sent by registered post, "
                "{post_days} after posting; and if sent by email, at the time of transmission provided no bounce "
                "message is received.",
                "A notice sent by registered post is deemed received {post_days} after posting, and a notice sent by "
                "email is deemed received when sent, unless sent outside business hours, in which case it is deemed "
                "received on the next Business Day.",
            ),
        ),
    ),
    _t(
        "entire_agreement",
        "base",
        ("Entire Agreement", "Whole Agreement", "General", "Miscellaneous"),
        (
            (
                "This Agreement constitutes the entire agreement between the parties relating to its subject matter "
                "and supersedes all prior agreements, representations and understandings, whether written or oral.",
                "This Agreement, together with any schedules and statements of work, sets out the whole agreement "
                "between the parties and replaces all earlier discussions and documents.",
            ),
            (
                "Each party acknowledges that in entering into this Agreement it has not relied on any statement, "
                "representation or warranty not expressly set out in this Agreement.",
                "No variation of this Agreement shall be effective unless made in writing and signed by or on behalf "
                "of each party.",
            ),
        ),
        (
            (
                "This Agreement may be executed in counterparts, each of which is an original and which together "
                "constitute one instrument.",
                "Nothing in this Agreement creates a partnership, joint venture or agency between the parties.",
            ),
        ),
    ),
    _t(
        "severability",
        "base",
        ("Severability", "Invalidity", "Severance"),
        (
            (
                "If any provision of this Agreement is held to be invalid, illegal or unenforceable, that provision "
                "shall be deemed modified to the minimum extent necessary to make it valid, and the remaining "
                "provisions shall continue in full force and effect.",
                "The invalidity or unenforceability of any provision of this Agreement shall not affect the validity "
                "or enforceability of the remaining provisions.",
            ),
        ),
        (
            (
                "No failure or delay by a party in exercising any right or remedy shall constitute a waiver of that "
                "right or remedy.",
                "The parties shall negotiate in good faith to replace any severed provision with a valid provision "
                "of similar effect.",
            ),
        ),
    ),
    _t(
        "force_majeure",
        "base",
        ("Force Majeure", "Events Beyond Control", "Excused Performance"),
        (
            (
                "Neither party shall be liable for any failure or delay in performing its obligations under this "
                "Agreement to the extent that such failure or delay is caused by an event beyond its reasonable "
                "control, including acts of God, flood, fire, epidemic, war, terrorism, strikes or failure of public "
                "utilities (a Force Majeure Event).",
                "A party shall not be in breach of this Agreement if it is prevented from performing by circumstances "
                "beyond its reasonable control, such as natural disaster, civil unrest, government action or "
                "industrial dispute.",
            ),
            (
                "The affected party shall notify the other party promptly of the Force Majeure Event and shall use "
                "reasonable endeavours to mitigate its effects. If the Force Majeure Event continues for more than "
                "{fm_days}, either party may terminate this Agreement by written notice.",
                "If performance is suspended for more than {fm_days} because of such an event, the party not "
                "affected may terminate this Agreement on written notice.",
            ),
        ),
    ),
    _t(
        "insurance",
        "base",
        ("Insurance", "Insurance Cover", "Insurance Requirements"),
        (
            (
                "{A} shall maintain in force, with a reputable insurer, professional indemnity insurance with a limit "
                "of not less than {insurance} for each and every claim for the duration of this Agreement and for "
                "{insurance_years} thereafter.",
                "{B} shall maintain comprehensive insurance for the contents of the Premises and public liability "
                "insurance of at least {insurance} and shall provide evidence of cover to {A} on request.",
                "Each party shall maintain insurance appropriate to its obligations under this Agreement, including "
                "public liability cover of not less than {insurance}.",
            ),
        ),
        (
            (
                "Evidence of the required insurance shall be provided to the other party on request.",
                "Failure to maintain the required insurance shall be a material breach of this Agreement.",
            ),
        ),
    ),
    _t(
        "data_protection",
        "base",
        ("Data Protection", "Data Privacy", "Processing of Personal Data", "Customer Data"),
        (
            (
                "{A} shall process Customer Data only on the documented instructions of {B} and shall implement "
                "appropriate technical and organisational measures to protect Customer Data against unauthorised or "
                "unlawful processing, accidental loss, destruction or damage.",
                "Each party shall comply with applicable data protection laws in respect of personal data processed "
                "under this Agreement, and {A} shall act only as a processor on behalf of {B}.",
            ),
            (
                "{A} shall notify {B} without undue delay, and in any event within {breach_hours} of becoming aware, "
                "of any personal data breach affecting Customer Data.",
                "In the event of a security incident affecting Customer Data, {A} shall inform {B} within "
                "{breach_hours} and shall cooperate with {B} in investigating and remedying the incident.",
            ),
        ),
        (
            (
                "Customer Data shall be hosted in data centres located in {data_region} and shall not be transferred "
                "outside that region without {B}'s prior written consent.",
                "On termination, {A} shall return or delete all Customer Data in accordance with {B}'s instructions.",
            ),
        ),
        contract_types=("saas",),
    ),
    _t(
        "services",
        "base",
        ("Services", "Scope of Services", "Provision of Services"),
        (
            (
                "{A} shall provide the services described in each statement of work executed by the parties (the "
                "Services) in accordance with this Agreement.",
                "{A} agrees to perform the consulting and development services set out in the applicable statement "
                "of work, and {B} agrees to provide the cooperation and information reasonably required.",
            ),
            (
                "{A} shall perform the Services with reasonable skill and care and shall allocate suitably qualified "
                "personnel.",
                "Any change to the scope of the Services shall be agreed in writing through the change control "
                "procedure set out in the applicable statement of work.",
            ),
        ),
        contract_types=("msa",),
    ),
    _t(
        "payment",
        "base",
        ("Fees and Payment", "Payment Terms", "Charges", "Fees"),
        (
            (
                "{B} shall pay {A} the fees set out in the applicable statement of work, which for the initial term "
                "amount to {fee} per annum, exclusive of applicable taxes.",
                "In consideration of the services, {B} shall pay {A} a fee of {fee} per year, payable "
                "{payment_frequency} in advance.",
                "The subscription fee for the services is {fee} per annum, exclusive of goods and services tax and "
                "any other applicable taxes.",
            ),
            (
                "{A} shall invoice {B} {payment_frequency} and {B} shall pay each undisputed invoice within "
                "{payment_days} of receipt.",
                "All invoices are payable within {payment_days} of the invoice date. Late payments shall bear "
                "interest at {interest_rate} per annum from the due date until payment in full.",
            ),
        ),
        (
            (
                "{B} may withhold payment of any amount disputed in good faith, provided it notifies {A} of the "
                "dispute before the due date.",
                "{A} may suspend the services if any undisputed invoice remains unpaid after the due date.",
            ),
        ),
        contract_types=("msa", "saas"),
    ),
    _t(
        "subscription",
        "base",
        ("Subscription", "Access and Use", "Licence Grant", "Provision of the Service"),
        (
            (
                "Subject to the terms of this Agreement, {A} grants {B} a non-exclusive, non-transferable right to "
                "access and use the Service during the subscription term for its internal business operations.",
                "{A} shall make the Service available to {B} through the internet via the login credentials issued "
                "to {B}'s authorised users, up to the number of users specified in the order form.",
            ),
            (
                "{B} shall not sublicense, resell or make the Service available to any third party except as "
                "expressly permitted in this Agreement.",
                "{B} is responsible for maintaining the confidentiality of its users' credentials and for all "
                "activity under its account.",
            ),
        ),
        contract_types=("saas",),
    ),
    _t(
        "acceptable_use",
        "base",
        ("Acceptable Use", "Restrictions", "Customer Obligations", "Use Restrictions"),
        (
            (
                "{B} shall not use the Service to store or transmit any material that is unlawful, defamatory, "
                "infringing or harmful, or to introduce any virus, malware or other harmful code.",
                "{B} shall not attempt to gain unauthorised access to the Service or its related systems, reverse "
                "engineer the Service, or use it to build a competing product.",
            ),
            (
                "{A} may suspend {B}'s access to the Service immediately if {A} reasonably believes that {B} is in "
                "breach of this clause, and shall restore access once the breach has been remedied.",
                "{B} shall ensure that its users comply with this clause and shall be responsible for any breach by "
                "its users.",
            ),
        ),
        contract_types=("saas",),
    ),
    _t(
        "sla",
        "base",
        ("Service Levels", "Service Level Agreement", "Availability", "Support and Service Levels"),
        (
            (
                "{A} shall use commercially reasonable efforts to make the Service available with a monthly uptime of "
                "at least {sla}, excluding scheduled maintenance notified at least {maintenance_notice} in advance.",
                "{A} commits to an availability of {sla} per calendar month for the Service, measured at the "
                "boundary of {A}'s hosting environment.",
            ),
            (
                "If availability in any calendar month falls below {sla}, {B} shall be entitled to a service credit "
                "of {credit_pct} of the monthly fee for that month, claimed in writing within {claim_days} of the end "
                "of the month.",
                "{A} shall provide support by email and telephone during business hours and shall respond to "
                "priority one incidents within {response_hours}.",
            ),
        ),
        contract_types=("saas",),
    ),
    _t(
        "duties",
        "base",
        ("Position and Duties", "Duties", "Role", "Appointment"),
        (
            (
                "{A} employs {B} in the position of {job_title}, reporting to the {manager_title}. {B} shall perform "
                "the duties reasonably assigned by {A} from time to time consistent with that position.",
                "{B} is appointed as {job_title} and shall report to the {manager_title} or such other person as {A} "
                "may nominate.",
            ),
            (
                "{B} shall devote the whole of their working time, attention and skill to the business of {A} and "
                "shall not engage in any other employment without the prior written consent of {A}.",
                "{B}'s normal place of work shall be {A}'s office in {city}, although {B} may be required to travel "
                "and to work from other locations as reasonably required.",
            ),
        ),
        contract_types=("employment",),
    ),
    _t(
        "compensation",
        "base",
        ("Remuneration", "Salary", "Compensation", "Pay"),
        (
            (
                "{A} shall pay {B} a gross annual salary of {salary}, payable in equal monthly instalments in arrears, "
                "less applicable deductions.",
                "{B} shall receive a base salary of {salary} per annum, reviewed annually, subject to statutory "
                "deductions for income tax and provident fund contributions.",
            ),
            (
                "{B} may be eligible for a discretionary performance bonus of up to {bonus_pct} of base salary, "
                "subject to the achievement of targets set by {A}.",
                "Salary reviews shall take place annually; any increase is at the sole discretion of {A}.",
            ),
        ),
        contract_types=("employment",),
    ),
    _t(
        "benefits",
        "base",
        ("Benefits", "Leave and Benefits", "Holidays and Benefits"),
        (
            (
                "{B} shall be entitled to {leave_days} of paid annual leave per year in addition to public holidays, "
                "to be taken at times agreed with {A}.",
                "{B} is entitled to paid annual leave of {leave_days} per calendar year, accruing pro rata, in "
                "addition to the public holidays observed by {A}.",
            ),
            (
                "{A} shall provide {B} with group medical insurance cover for {B} and their immediate family in "
                "accordance with {A}'s policies as amended from time to time.",
                "{B} shall participate in the employees' provident fund scheme and such other benefit schemes as {A} "
                "may operate from time to time.",
            ),
        ),
        contract_types=("employment",),
    ),
    _t(
        "premises",
        "base",
        ("Premises", "Demise", "Grant of Lease", "Permitted Use"),
        (
            (
                "{A} hereby lets to {B} the premises known as {premises_address} (the Premises) together with the "
                "right to use the common areas of the building for access.",
                "{A} agrees to let and {B} agrees to take the Premises at {premises_address}, including the fixtures "
                "and fittings listed in the inventory signed by the parties.",
            ),
            (
                "{B} shall use the Premises only for {permitted_use} and for no other purpose without the prior "
                "written consent of {A}.",
                "The Premises shall be used for {permitted_use} only, and {B} shall comply with all laws and "
                "building rules applicable to that use.",
            ),
        ),
        contract_types=("lease",),
    ),
    _t(
        "rent",
        "base",
        ("Rent", "Rent and Payment", "Rental"),
        (
            (
                "{B} shall pay {A} rent of {rent} per month, payable in advance on or before the {rent_day} day of "
                "each calendar month by bank transfer to the account nominated by {A}.",
                "The monthly rent shall be {rent}, exclusive of utilities, maintenance charges and applicable taxes, "
                "payable monthly in advance.",
            ),
            (
                "The rent shall increase by {escalation} on each anniversary of the commencement date.",
                "Any rent not paid within {grace_days} of the due date shall bear interest at {interest_rate} per "
                "annum until paid.",
            ),
        ),
        contract_types=("lease",),
    ),
    _t(
        "security_deposit",
        "base",
        ("Security Deposit", "Deposit"),
        (
            (
                "{B} shall pay {A} an interest-free security deposit of {deposit} on or before the commencement date, "
                "to be held by {A} as security for the performance of {B}'s obligations under this Lease.",
                "On signing this Lease {B} shall deposit with {A} the sum of {deposit} as security, which {A} may "
                "apply against unpaid rent or the cost of repairing damage caused by {B}.",
            ),
            (
                "The deposit shall be refunded to {B} within {deposit_refund_days} of the expiry or earlier "
                "termination of this Lease, less any amounts lawfully deducted.",
                "{A} shall return the deposit, without interest, within {deposit_refund_days} after {B} vacates the "
                "Premises and returns the keys.",
            ),
        ),
        contract_types=("lease",),
    ),
    _t(
        "maintenance",
        "base",
        ("Maintenance and Repairs", "Repairs", "Condition of Premises"),
        (
            (
                "{B} shall keep the interior of the Premises in good and tenantable condition and shall carry out "
                "minor repairs at its own cost.",
                "{A} shall be responsible for structural repairs and for the maintenance of the roof, external walls "
                "and common services of the building.",
            ),
            (
                "{B} shall permit {A} to enter the Premises on {inspection_notice} prior notice to inspect their "
                "condition and to carry out repairs.",
                "{B} shall not make any structural alterations to the Premises without the prior written consent "
                "of {A}.",
            ),
        ),
        contract_types=("lease",),
    ),
    _t(
        "return_of_materials",
        "base",
        ("Return of Information", "Return or Destruction", "Return of Materials"),
        (
            (
                "Upon the written request of {A}, or on termination of this Agreement, {B} shall promptly return or "
                "destroy all Confidential Information and all copies, notes and extracts, and shall certify in "
                "writing that it has done so within {return_days}.",
                "Within {return_days} of a request by {A}, {B} shall return all documents and materials containing "
                "Confidential Information and shall permanently erase all electronic copies.",
            ),
        ),
        (
            (
                "{B} may retain one archival copy of the Confidential Information to the extent required by law, "
                "subject to the continuing obligations of confidentiality.",
                "The return or destruction of Confidential Information shall not release {B} from its obligations "
                "under this Agreement.",
            ),
        ),
        contract_types=("nda",),
    ),
    _t(
        "remedies",
        "base",
        ("Remedies", "Injunctive Relief", "Equitable Relief"),
        (
            (
                "{B} acknowledges that damages alone may not be an adequate remedy for breach of this Agreement and "
                "that {A} shall be entitled to seek injunctive relief or specific performance in addition to any "
                "other remedy available at law or in equity.",
                "{B} agrees that any breach of this Agreement may cause irreparable harm to {A} for which monetary "
                "damages would be inadequate, and that {A} may seek an injunction without the need to post a bond.",
            ),
        ),
        (
            (
                "The rights and remedies provided in this Agreement are cumulative and not exclusive of any rights "
                "or remedies provided by law.",
                "{B} shall reimburse {A} for the reasonable legal costs of enforcing this Agreement.",
            ),
        ),
        contract_types=("nda",),
    ),
]


def templates_for(label: str, contract_type: str) -> list[ClauseTemplate]:
    """All template variants for ``label`` that apply to ``contract_type``."""
    return [
        t for t in TEMPLATES if t.label == label and (t.contract_types is None or contract_type in t.contract_types)
    ]


ALL_LABELS: tuple[str, ...] = tuple(sorted({t.label for t in TEMPLATES} | {"preamble", "recitals", "signature"}))
