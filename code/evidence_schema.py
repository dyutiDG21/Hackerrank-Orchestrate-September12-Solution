"""Validated structured evidence claims for Buy or Wait? extraction.

This module validates model-produced JSON before any downstream financial logic
can consume it. It does not apply claims to events or resolve conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Mapping

from data_layer import DataValidationError, parse_date, parse_decimal


SCHEMA_VERSION = "evidence_claims_v2"

SOURCE_TYPES = {"message", "image"}
CLAIM_TYPES = {
    "missing_amount",
    "amount_confirmation",
    "amount_amendment",
    "date_confirmation",
    "date_amendment",
    "status_confirmation",
    "cancellation",
    "invalidation",
    "payroll_change",
    "salary_change",
    "pending_payment_clarification",
    "future_payment_clarification",
}

DOCUMENT_ROLES = {
    "base_salary",
    "total_earnings",
    "total_deductions",
    "net_pay",
    "transferred_amount",
    "total_amount",
    "amount_paid",
    "amount_received",
    "balance_due",
    "outstanding_amount",
}

IDENTIFIER_ROLES = {
    "receipt_number",
    "employee_number",
    "payroll_reference",
}


class EvidenceValidationError(DataValidationError):
    """Raised when extracted evidence is malformed or outside the allowed schema."""


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    source_type: str
    source_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    claim_type: str
    document_role: str | None
    amount: Decimal | None
    currency: str | None
    effective_date: date | None
    status: str | None
    identifier_role: str | None
    referenced_external_id: str | None
    provenance: str
    raw_claim: Mapping[str, Any]


@dataclass(frozen=True)
class EvidenceExtractionResult:
    source_type: str
    source_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    schema_version: str
    claims: tuple[EvidenceClaim, ...]
    raw_response: Mapping[str, Any]


def evidence_response_json_schema() -> dict[str, Any]:
    claim_properties: dict[str, Any] = {
        "claim_id": {"type": "string"},
        "claim_type": {"type": "string", "enum": sorted(CLAIM_TYPES)},
        "document_role": {"type": ["string", "null"], "enum": sorted(DOCUMENT_ROLES) + [None]},
        "amount": {"type": ["string", "number", "null"]},
        "currency": {"type": ["string", "null"]},
        "effective_date": {"type": ["string", "null"], "description": "YYYY-MM-DD when present"},
        "status": {"type": ["string", "null"]},
        "identifier_role": {"type": ["string", "null"], "enum": sorted(IDENTIFIER_ROLES) + [None]},
        "referenced_external_id": {"type": ["string", "null"]},
        "provenance": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "claims"],
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(claim_properties),
                    "properties": claim_properties,
                },
            },
        },
    }


def optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvidenceValidationError(f"{field} must be a string or null")
    text = value.strip()
    return text or None


def required_text(value: Any, field: str) -> str:
    text = optional_text(value, field)
    if text is None:
        raise EvidenceValidationError(f"{field} is required")
    return text


def parse_optional_decimal(value: Any, field: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise EvidenceValidationError(f"{field} must be a decimal-compatible value or null")
    return parse_decimal(str(value), field=field)


def parse_optional_date(value: Any, field: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvidenceValidationError(f"{field} must be YYYY-MM-DD or null")
    return parse_date(value, field=field)


def validate_claim(
    raw_claim: Mapping[str, Any],
    *,
    source_type: str,
    source_id: str,
    user_id: str,
    request_id: str | None,
    related_event_id: str | None,
) -> EvidenceClaim:
    if not isinstance(raw_claim, Mapping):
        raise EvidenceValidationError("Each claim must be an object")
    claim_type = required_text(raw_claim.get("claim_type"), "claim_type")
    if claim_type not in CLAIM_TYPES:
        raise EvidenceValidationError(f"Unsupported claim_type: {claim_type}")
    document_role = optional_text(raw_claim.get("document_role"), "document_role")
    if document_role is not None and document_role not in DOCUMENT_ROLES:
        raise EvidenceValidationError(f"Unsupported document_role: {document_role}")
    identifier_role = optional_text(raw_claim.get("identifier_role"), "identifier_role")
    if identifier_role is not None and identifier_role not in IDENTIFIER_ROLES:
        raise EvidenceValidationError(f"Unsupported identifier_role: {identifier_role}")
    amount = parse_optional_decimal(raw_claim.get("amount"), "amount")
    currency = optional_text(raw_claim.get("currency"), "currency")
    if amount is None and currency is not None:
        raise EvidenceValidationError("currency cannot be present when amount is null")
    if amount is not None and currency is None:
        raise EvidenceValidationError("currency is required when amount is present")
    return EvidenceClaim(
        claim_id=required_text(raw_claim.get("claim_id"), "claim_id"),
        source_type=source_type,
        source_id=source_id,
        user_id=user_id,
        request_id=request_id,
        related_event_id=related_event_id,
        claim_type=claim_type,
        document_role=document_role,
        amount=amount,
        currency=currency,
        effective_date=parse_optional_date(raw_claim.get("effective_date"), "effective_date"),
        status=optional_text(raw_claim.get("status"), "status"),
        identifier_role=identifier_role,
        referenced_external_id=optional_text(raw_claim.get("referenced_external_id"), "referenced_external_id"),
        provenance=required_text(raw_claim.get("provenance"), "provenance"),
        raw_claim=dict(raw_claim),
    )


def exact_claim_key(claim: EvidenceClaim) -> tuple[object, ...]:
    return (
        claim.source_type,
        claim.source_id,
        claim.user_id,
        claim.request_id,
        claim.related_event_id,
        claim.claim_type,
        claim.document_role,
        claim.amount,
        claim.currency,
        claim.effective_date,
        claim.status,
        claim.identifier_role,
        claim.referenced_external_id,
        claim.provenance,
    )


def cleanup_claims(claims: tuple[EvidenceClaim, ...]) -> tuple[EvidenceClaim, ...]:
    deduped: list[EvidenceClaim] = []
    seen_exact: set[tuple[object, ...]] = set()
    for claim in claims:
        key = exact_claim_key(claim)
        if key in seen_exact:
            continue
        seen_exact.add(key)
        deduped.append(claim)

    role_specific_amounts = {
        (claim.amount, claim.currency)
        for claim in deduped
        if claim.document_role is not None and claim.amount is not None and claim.currency is not None
    }
    cleaned = [
        claim
        for claim in deduped
        if not (
            claim.document_role is None
            and claim.amount is not None
            and claim.currency is not None
            and (claim.amount, claim.currency) in role_specific_amounts
        )
    ]
    return tuple(cleaned)


def validate_evidence_response(
    response: Mapping[str, Any],
    *,
    source_type: str,
    source_id: str,
    user_id: str,
    request_id: str | None,
    related_event_id: str | None,
) -> EvidenceExtractionResult:
    if source_type not in SOURCE_TYPES:
        raise EvidenceValidationError(f"Unsupported source_type: {source_type}")
    if not isinstance(response, Mapping):
        raise EvidenceValidationError("Evidence response must be a JSON object")
    schema_version = required_text(response.get("schema_version"), "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise EvidenceValidationError(f"Unsupported schema_version: {schema_version}")
    raw_claims = response.get("claims")
    if not isinstance(raw_claims, list):
        raise EvidenceValidationError("claims must be a list")
    claims = cleanup_claims(tuple(
        validate_claim(
            claim,
            source_type=source_type,
            source_id=source_id,
            user_id=user_id,
            request_id=request_id,
            related_event_id=related_event_id,
        )
        for claim in raw_claims
    ))
    return EvidenceExtractionResult(
        source_type=source_type,
        source_id=source_id,
        user_id=user_id,
        request_id=request_id,
        related_event_id=related_event_id,
        schema_version=schema_version,
        claims=claims,
        raw_response=dict(response),
    )


def result_to_jsonable(result: EvidenceExtractionResult) -> dict[str, Any]:
    return {
        "source_type": result.source_type,
        "source_id": result.source_id,
        "user_id": result.user_id,
        "request_id": result.request_id,
        "related_event_id": result.related_event_id,
        "schema_version": result.schema_version,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "document_role": claim.document_role,
                "amount": str(claim.amount) if claim.amount is not None else None,
                "currency": claim.currency,
                "effective_date": claim.effective_date.isoformat() if claim.effective_date is not None else None,
                "status": claim.status,
                "identifier_role": claim.identifier_role,
                "referenced_external_id": claim.referenced_external_id,
                "provenance": claim.provenance,
            }
            for claim in result.claims
        ],
    }
