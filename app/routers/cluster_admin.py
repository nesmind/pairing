"""Cluster page (admin-only). Logic lives in app.services.cluster_service."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal, get_db
from app.models import User
from app.schemas import MatricxonCachePersistence, MatricxonCachePersistenceUpdate, OkResponse
from app.schemas.cluster import (
    AdminSecret,
    ClusterCache,
    ClusterMap,
    ClusterPaths,
    ClusterResetResult,
    ClusterState,
    ClusterTimers,
    ClusterUpdate,
    JoinCluster,
    MapNode,
    NodeAdd,
    NodeAddress,
    NodeCheck,
    PeerList,
    QuorumState,
    VoterId,
)
from app.services.auth_service import require_admin
from app.services.cluster_join import ClusterJoin
from app.services.cluster_map import ClusterMapBuilder
from app.services.cluster_nodes import (
    ClusterNodes,
    NodeNotAllowedError,
    NodeNotFreshError,
    NodeResetRefusedError,
)
from app.services.cluster_peers import ClusterPeers
from app.services.cluster_reset import ClusterReset
from app.services.cluster_secret import ClusterSecret
from app.services.cluster_service import ClusterService
from app.services.cluster_settings import ClusterSettings, ConsensusOffError
from app.services.matricxon_cache_persistence import MatricxonCachePersistenceService
from app.services.matricxon_client import MatricxonError

router = APIRouter(prefix="/api/cluster", tags=["cluster"])


@router.get("", response_model=ClusterState)
async def get_cluster(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)) -> ClusterState:
    return await ClusterService(db, with_fresh=True).state()


@router.get("/map", response_model=ClusterMap)
async def get_map(
    compact: bool = False, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterMap:
    """The Nodes map; `compact` drops per-node details for big clusters."""
    return await ClusterMapBuilder(db, ClusterService(db)).build(compact=compact)


@router.get("/map/node", response_model=MapNode)
async def get_map_node(id: str, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)) -> MapNode:
    """One node's full record, for a map loaded compact."""
    node = await ClusterMapBuilder(db, ClusterService(db)).node(id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"No node {id} in this cluster.")
    return node


@router.put("", response_model=ClusterState)
async def set_cluster(
    body: ClusterUpdate, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterState:
    try:
        return await ClusterService(db).set_enabled(body.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/leave", response_model=ClusterState)
async def leave_cluster(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)) -> ClusterState:
    """Cluster mode off for this pAIring only; nodes stay listed as plain remote hosts."""
    await ClusterJoin(db).leave()
    return await ClusterService(db).state()


@router.post("/join", response_model=ClusterState)
async def join_cluster(
    body: JoinCluster, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterState:
    """Joins an existing cluster by any node's address and the cluster id."""
    try:
        await ClusterJoin(db).join(body.address, body.cluster_id, body.admin_secret.strip())
    except NodeNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await ClusterService(db).state()


@router.put("/paths", response_model=ClusterState)
async def set_cluster_paths(
    body: ClusterPaths, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterState:
    await _require_quorum(db)
    try:
        return await ClusterService(db).set_paths(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _require_quorum(db: AsyncSession) -> None:
    """Member changes need a voter majority; otherwise they only time out."""
    if (await ClusterService(db).state()).quorum == "lost":
        raise HTTPException(
            status_code=409, detail="No quorum: a majority of the voters is not reachable. Restore it first."
        )


@router.get("/quorum", response_model=QuorumState)
async def get_quorum(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)) -> QuorumState:
    """Whether the cluster has a quorum (one cheap status call per node)."""
    return QuorumState(quorum=ClusterService.quorum_of(await ClusterSettings(db).voters_seen()))


@router.put("/cache", response_model=MatricxonCachePersistence)
async def set_cache(
    body: ClusterCache, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> MatricxonCachePersistence:
    """Saved-chat-cache limits for every node, through the cluster (needs a quorum).
    Without consensus, each node is told directly.
    """
    await _require_quorum(db)
    try:
        problem = await ClusterSettings(db).set_cache(body)
        if problem:
            raise HTTPException(status_code=502, detail=problem)
        return await MatricxonCachePersistenceService(shared_folder=True).status()
    except ConsensusOffError:
        return await MatricxonCachePersistenceService(shared_folder=True).update(
            MatricxonCachePersistenceUpdate(**body.model_dump())
        )
    except MatricxonError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.put("/timers", response_model=ClusterState)
async def set_timers(
    body: ClusterTimers, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterState:
    """Sets the consensus timers on every voter (needs a quorum)."""
    await _require_quorum(db)
    try:
        note = await ClusterSettings(db).set_timers(body)
    except ConsensusOffError as exc:
        raise HTTPException(status_code=409, detail="The cluster is not running consensus yet.") from exc
    return await ClusterService(db).state(note)


@router.post("/activate", response_model=ClusterState)
async def activate_cluster(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)) -> ClusterState:
    await _require_quorum(db)
    try:
        return await ClusterService(db).activate()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/nodes/check", response_model=NodeCheck)
async def check_node(
    body: NodeAddress, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> NodeCheck:
    try:
        return await ClusterNodes(db).check(body.address)
    except NodeNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/peers", response_model=PeerList)
async def get_peers(
    id: str,
    q: str = "",
    offset: int = 0,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> PeerList:
    """One node's address book, paged and searchable (debugging)."""
    try:
        return await ClusterPeers(db).of(id, q, offset, limit)
    except NodeNotAllowedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/nodes/demote", response_model=OkResponse)
async def demote_node(
    body: NodeAddress, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> OkResponse:
    """Makes a voter a learner. Refused for the last voter; needs a quorum."""
    await _require_quorum(db)
    try:
        await ClusterNodes(db).demote(body.address)
    except NodeNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OkResponse(ok=True)


@router.post("/reset", response_model=ClusterResetResult)
async def reset_cluster(
    db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterResetResult:
    """Resets every node to a fresh install and cluster mode off. No quorum needed;
    shared folders are untouched. Unreachable nodes are listed in `failed`.
    """
    return await ClusterReset(db).run()


@router.post("/reset/stream", response_model=None)
async def reset_cluster_stream(_admin: User = Depends(require_admin)) -> StreamingResponse:
    """Same reset as server-sent events: `start`, one `node` per node, then `done`."""

    async def events():
        async with AsyncSessionLocal() as db:  # own session: the request's closes before the stream ends
            async for event in ClusterReset(db).stream():
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.post("/nodes/reset", response_model=OkResponse)
async def reset_node(
    body: NodeAddress, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> OkResponse:
    """Resets a removed node to a fresh install (it restarts) so it can be re-added."""
    try:
        await ClusterNodes(db).reset(body.address)
    except NodeNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NodeResetRefusedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return OkResponse(ok=True)


@router.post("/nodes", response_model=ClusterState)
async def add_node(
    body: NodeAdd, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterState:
    """Checks the node is a fresh Matricxon, configures it as a joiner and adds it to the pool.
    `advertise_url`: how other nodes reach it, if different from `address`.
    """
    await _require_quorum(db)
    try:
        await ClusterNodes(db).add(body.address, body.advertise_url, body.node_kind)
    except NodeNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NodeNotFreshError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return await ClusterService(db).state()


@router.delete("/voters", response_model=ClusterState)
async def remove_unknown_voter(
    body: VoterId, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterState:
    """Removes a voter that is not one of pAIring's nodes. Needs a quorum."""
    await _require_quorum(db)
    try:
        await ClusterNodes(db).drop_unknown_voter(body.id)
    except NodeNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await ClusterService(db).state()


@router.delete("/nodes", response_model=ClusterState)
async def remove_node(
    body: NodeAddress, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterState:
    await _require_quorum(db)
    try:
        await ClusterNodes(db).remove(body.address)
    except NodeNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await ClusterService(db).state()


@router.put("/admin-secret", response_model=ClusterState)
async def set_admin_secret(
    body: AdminSecret, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterState:
    """Sets the secret pAIring signs its admin calls to the nodes with ("" removes it). All nodes must answer."""
    try:
        await ClusterSecret(db).set(body.secret.strip())
    except NodeNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await ClusterService(db).state()
