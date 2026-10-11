"""Cluster page: service (mocked hosts), router on/off, page route and sidebar link."""

import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import RedirectResponse
from starlette.requests import Request

from app.routers import cluster_admin, pages
from app.schemas.cluster import ClusterCheck, ClusterPaths, ClusterUpdate
from app.services import engine_service, image_engine_service, matricxon_pool, matricxon_process, settings_service
from app.services.cluster_service import ClusterService

A, B = "http://a:8420", "http://b:8420"
_REAL_PUSH = ClusterService._push_to_seed


@pytest.fixture(autouse=True)
def _quorum_ok(monkeypatch):
    monkeypatch.setattr(ClusterService, "quorum_of", staticmethod(lambda hosts: "ok"))


@pytest.fixture(autouse=True)
def _never_configure_the_real_seed(monkeypatch):
    """Saving cluster settings makes the seed rewrite its .env and restart; tests must never reach
    the live Matricxon. Push tests bring their own transport (see `_REAL_PUSH`).
    """

    async def no_push(self, config, target=None):
        return None

    monkeypatch.setattr(ClusterService, "_push_to_seed", no_push)


def _readiness(role: str = "off", ready: bool = False) -> dict:
    check = {"id": "prompt_cache", "status": "ok" if ready else "action_needed", "what": "w", "why": "y"}
    return {"role": role, "node_id": "n1", "ready": ready, "active": False, "checks": [check], "nodes": []}


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def _first(db, first: str, hosts: tuple[str, ...] = (), added: tuple[str, ...] = ()) -> None:
    """Saves `first` as the starting node; `hosts` are the plain hosts (default: only it)."""
    config = await settings_service.get_matricxon_server_config(db)
    config.mode, config.remote_hosts, config.cluster_hosts = "remote", list(hosts or (first,)), list(added)
    config.first_node = first
    await settings_service.set_matricxon_server_config(db, config)
    matricxon_pool.refresh_from_config(config)


@pytest.fixture
async def matricxon_engine(db):
    await engine_service.set_active_engine(db, "matricxon")


@pytest.mark.asyncio
async def test_state_lists_each_hosts_checklist(db, matricxon_engine):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "b":
            return httpx.Response(404)
        return httpx.Response(200, json=_readiness())

    state = await ClusterService(db, [A, B], _transport(handler)).state()

    assert state.supported and not state.enabled and not state.restart_needed
    good, old = state.hosts
    assert good.reachable and good.checks[0].status == "action_needed" and good.node_id == "n1"
    assert not old.reachable and "no cluster mode" in old.error


@pytest.mark.asyncio
async def test_unreachable_host_is_reported_not_raised(db, matricxon_engine):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    state = await ClusterService(db, [A], _transport(handler)).state()
    assert state.hosts[0].error == "refused" and not state.hosts[0].reachable


@pytest.mark.asyncio
async def test_turning_on_saves_the_seed_role_and_asks_for_a_restart(db, matricxon_engine, monkeypatch):
    written: list[str] = []
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: written.append(config.cluster_role))
    config = await settings_service.get_matricxon_server_config(db)
    config.first_node = matricxon_pool.LOCAL_MATRICXON_HOST
    await settings_service.set_matricxon_server_config(db, config)
    service = ClusterService(db, [A], _transport(lambda r: httpx.Response(200, json=_readiness("off"))))

    state = await service.set_enabled(True)

    assert state.enabled and state.restart_needed
    assert written == ["seed"]
    assert (await settings_service.get_matricxon_server_config(db)).cluster_role == "seed"
    service = ClusterService(db, [A], _transport(lambda r: httpx.Response(200, json=_readiness("seed", True))))
    assert not (await service.state()).restart_needed


@pytest.mark.asyncio
async def test_turning_off_clears_the_role(db, matricxon_engine, monkeypatch):
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    await _first(db, A)
    service = ClusterService(db, [A], _transport(lambda r: httpx.Response(200, json=_readiness("off"))))
    await service.set_enabled(True)
    assert not (await service.set_enabled(False)).enabled
    assert (await settings_service.get_matricxon_server_config(db)).cluster_role == "off"


@pytest.mark.asyncio
async def test_ollama_engine_is_not_supported_and_asks_no_host(db):
    await engine_service.set_active_engine(db, "ollama")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no host should be asked")

    state = await ClusterService(db, [A], _transport(handler)).state()
    assert not state.supported and state.hosts == []


@pytest.mark.asyncio
async def test_router_sets_the_role(db, matricxon_engine, monkeypatch):
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    await _first(db, A)
    state = await cluster_admin.set_cluster(ClusterUpdate(enabled=True), db=db, _admin=None)
    assert state.enabled
    assert (await cluster_admin.get_cluster(db=db, _admin=None)).enabled


def _request(user_id: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/cluster",
            "headers": [],
            "query_string": b"",
            "session": {"user_id": user_id},
        }
    )


@pytest.mark.asyncio
async def test_cluster_page_is_admin_only(db, user, admin_user):
    assert isinstance(await pages.cluster_page(_request(user.id), db=db), RedirectResponse)
    page = await pages.cluster_page(_request(admin_user.id), db=db)
    html = page.body.decode()
    assert 'id="cluster-switch"' in html and "cluster.js" in html


@pytest.mark.asyncio
async def test_sidebar_shows_cluster_link_to_admins_only(db, user, admin_user):
    for who, expected in ((admin_user, True), (user, False)):
        response = await pages.chat_page(_request(who.id), db=db)
        assert ('href="/cluster"' in response.body.decode()) is expected


@pytest.mark.asyncio
async def test_set_paths_saves_them_to_the_config_and_env_and_blank_means_default(db, matricxon_engine, monkeypatch):
    written: list[tuple] = []
    monkeypatch.setattr(
        matricxon_process, "write_env_file", lambda c: written.append((c.models_path, c.prompt_cache_path))
    )
    service = ClusterService(db, [], None)

    state = await service.set_paths(ClusterPaths(models_path=" /mnt/m ", prompt_cache_path="/mnt/c"))
    assert (state.paths.models_path, state.paths.prompt_cache_path) == ("/mnt/m", "/mnt/c")
    assert written[-1] == ("/mnt/m", "/mnt/c")

    state = await service.set_paths(ClusterPaths(models_path="", prompt_cache_path="  "))
    assert state.paths.models_path is None and state.paths.prompt_cache_path is None


def test_the_cache_folder_reaches_matricxons_environment(tmp_path):
    from app.schemas import MatricxonServerConfig

    config = MatricxonServerConfig(prompt_cache_path="/mnt/c")
    assert matricxon_process._build_env(config)["MATRICXON_PROMPT_CACHE_DIR"] == "/mnt/c"
    assert "MATRICXON_PROMPT_CACHE_DIR" not in matricxon_process._build_env(MatricxonServerConfig())


async def _mode_on(db) -> None:
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role = "seed"
    await settings_service.set_matricxon_server_config(db, config)


@pytest.mark.asyncio
async def test_activate_is_refused_while_cluster_mode_is_off_and_says_when_a_node_did_not_activate(
    db, matricxon_engine
):
    handler = lambda request: httpx.Response(200, json=_readiness("seed", True) | {"active": False})  # noqa: E731
    service = ClusterService(db, [A], _transport(handler), _FakeProbe(True))
    with pytest.raises(ValueError, match="Turn cluster mode on"):
        await service.activate()
    await _mode_on(db)
    state = await service.activate()  # answers but never becomes active (restarting)
    assert A in state.message and "did not activate" in state.message


@pytest.mark.asyncio
async def test_activate_asks_every_host_then_returns_the_state(db, matricxon_engine):
    await _mode_on(db)
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.host == "b":
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=_readiness("seed", True))

    state = await ClusterService(db, [A, B], _transport(handler), _FakeProbe(True)).activate()

    assert ("POST", "/api/cluster/activate") in seen
    assert sum(1 for method, _ in seen if method == "POST") == 2
    assert [h.reachable for h in state.hosts] == [True, False]


@pytest.mark.asyncio
async def test_router_saves_paths_and_activates(db, matricxon_engine, monkeypatch):
    await _mode_on(db)
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    state = await cluster_admin.set_cluster_paths(ClusterPaths(prompt_cache_path="/mnt/c"), db=db, _admin=None)
    assert state.paths.prompt_cache_path == "/mnt/c"
    assert (await cluster_admin.activate_cluster(db=db, _admin=None)).paths.prompt_cache_path == "/mnt/c"


class _FakeProbe:
    def __init__(self, shared: bool) -> None:
        self._shared = shared

    def is_shared(self, _path) -> bool:
        return self._shared

    def fstype(self, _path) -> str:
        return "nfs4" if self._shared else "ext4"


@pytest.mark.asyncio
async def test_a_local_images_folder_only_warns_and_never_blocks_activation(
    db, matricxon_engine, tmp_path, monkeypatch
):
    await _mode_on(db)
    monkeypatch.setattr("app.services.cluster_service.IMAGES_DIR", tmp_path)
    (tmp_path / "a.png").write_bytes(b"x" * 10)
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts.append(request.url.path)
        return httpx.Response(200, json=_readiness("seed", True))

    local = ClusterService(db, [A], _transport(handler), _FakeProbe(False))
    (check,) = (await local.state()).app_checks
    assert check.status == "action_needed" and check.size_bytes == 10 and "ext4" in check.hint
    await local.activate()
    assert posts == ["/api/cluster/activate"]

    shared = ClusterService(db, [A], _transport(handler), _FakeProbe(True))
    assert (await shared.state()).app_checks[0].status == "ok"


@pytest.mark.asyncio
async def test_saving_the_cluster_folders_leaves_the_image_engines_folder_alone(
    db, matricxon_engine, tmp_path, monkeypatch
):
    """The images folder belongs to the image engine's settings, not the cluster folders."""
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    folder = tmp_path / "imgs"
    folder.mkdir()
    sdcpp = await image_engine_service.get_sdcpp_config(db)
    sdcpp.images_path = str(folder)
    await image_engine_service.set_sdcpp_config(db, sdcpp)
    state = await ClusterService(db, [], None, _FakeProbe(True)).set_paths(ClusterPaths(models_path="/mnt/m"))
    assert not hasattr(state.paths, "images_path")
    assert (await image_engine_service.get_sdcpp_config(db)).images_path == str(folder)


async def _no_broadcast(_path: str) -> list[str]:
    return []


@pytest.mark.asyncio
async def test_an_added_joiner_does_not_make_the_seed_ask_for_a_restart(db, matricxon_engine, monkeypatch):
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role, config.cluster_hosts = "seed", [B]
    await settings_service.set_matricxon_server_config(db, config)

    def handler(request: httpx.Request) -> httpx.Response:
        role = "seed" if request.url.host == "a" else "joiner"
        return httpx.Response(200, json=_readiness(role, True))

    state = await ClusterService(db, [A, B], _transport(handler)).state()
    assert [h.role for h in state.hosts] == ["seed", "joiner"]
    assert state.restart_needed is False
    config.cluster_role = "off"
    await settings_service.set_matricxon_server_config(db, config)
    assert (await ClusterService(db, [A, B], _transport(handler)).state()).restart_needed is True


@pytest.mark.asyncio
async def test_the_cluster_page_has_three_views_and_the_add_form_lives_under_add_nodes(db, admin_user):
    page = await pages.cluster_page(_request(admin_user.id), db=db)
    html = page.body.decode()
    views = [html.index(f'id="cluster-view-{name}"') for name in ("overview", "nodes", "add")]
    assert views == sorted(views)
    overview, nodes, add = (html[views[0] : views[1]], html[views[1] : views[2]], html[views[2] :])
    assert 'id="cluster-node-form"' in add and 'id="cluster-node-form"' not in overview
    assert 'id="cluster-map"' in nodes and 'id="cluster-hosts"' in overview
    for name in ("overview", "nodes", "add"):
        assert f'data-view="{name}"' in html
    assert 'id="cluster-node-panel"' in html
    for script in ("cluster_node_panel.js", "cluster_map.js", "cluster.js"):
        assert f"/static/js/{script}" in html


def test_the_page_scripts_wire_the_map_to_the_nav_and_the_panel_actions():
    from pathlib import Path

    js = Path(__file__).parent.parent / "app" / "static" / "js"
    page, nodes, panel = (
        (js / "cluster.js").read_text(),
        (js / "cluster_map.js").read_text(),
        (js / "cluster_node_panel.js").read_text(),
    )
    assert 'clusterMap.setActive(name === "nodes")' in page  # map polls only while shown
    assert "/api/cluster/map" in nodes and "MAP_REFRESH_MS" in nodes
    assert "/api/cluster/activate" in panel and "/api/cluster/nodes" in panel
    assert "escapeHtml" in panel  # server text is escaped


@pytest.mark.asyncio
async def test_the_seeds_address_is_saved_and_reaches_matricxons_environment(db, matricxon_engine, monkeypatch):
    written: list = []
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda c: written.append(c.advertise_url))
    service = ClusterService(db, [], None)
    state = await service.set_paths(ClusterPaths(advertise_url=" http://10.0.0.5:8420/ "))
    assert state.paths.advertise_url == "http://10.0.0.5:8420" and written == ["http://10.0.0.5:8420"]
    # No page field for it any more: a save that omits it keeps it...
    assert (await service.set_paths(ClusterPaths(models_path="/m"))).paths.advertise_url == "http://10.0.0.5:8420"
    # ...only an explicit empty value (API) clears it.
    assert (await service.set_paths(ClusterPaths(advertise_url=""))).paths.advertise_url is None
    from app.schemas import MatricxonServerConfig

    env = matricxon_process._build_env(MatricxonServerConfig(advertise_url="http://x:1"))
    assert env["MATRICXON_ADVERTISE_URL"] == "http://x:1"
    assert "MATRICXON_ADVERTISE_URL" not in matricxon_process._build_env(MatricxonServerConfig())


def _seed_replies(status: int = 200, body: dict | None = None, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append((request.method, f"{request.url.host}", request.url.path, request.content))
        if request.url.path == "/api/cluster/configure":
            return httpx.Response(status, json=body or {"restarting": True})
        return httpx.Response(200, json=_readiness("seed", True))

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_saving_cluster_settings_tells_the_seed_to_apply_them_itself(db, matricxon_engine, monkeypatch):
    monkeypatch.setattr(ClusterService, "_push_to_seed", _REAL_PUSH)
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    seen: list = []
    await _first(db, A)
    service = ClusterService(db, [A], _seed_replies(seen=seen))

    state = await service.set_enabled(True)
    assert state.message is None
    configure = [s for s in seen if s[2] == "/api/cluster/configure"]
    assert json.loads(configure[0][3]) == {
        "role": "seed",
        "models_dir": "",
        "prompt_cache_dir": "",
        "advertise_url": None,  # nothing set: node keeps its own ("" would delete it)
    }
    await service.set_paths(ClusterPaths(models_path="/m", prompt_cache_path="/c", advertise_url="http://seed:8420/"))
    last = json.loads([s for s in seen if s[2] == "/api/cluster/configure"][-1][3])
    assert last == {"role": "seed", "models_dir": "/m", "prompt_cache_dir": "/c", "advertise_url": "http://seed:8420"}
    await service.set_enabled(False)
    assert json.loads([s for s in seen if s[2] == "/api/cluster/configure"][-1][3])["role"] == "off"


@pytest.mark.asyncio
async def test_the_seed_is_the_first_node_that_was_not_added_from_the_page(db, matricxon_engine, monkeypatch):
    monkeypatch.setattr(ClusterService, "_push_to_seed", _REAL_PUSH)
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    await _first(db, B, hosts=(B,), added=(A,))
    seen: list = []
    await ClusterService(db, [A, B], _seed_replies(seen=seen)).set_enabled(True)
    assert [s[1] for s in seen if s[2] == "/api/cluster/configure"] == ["b"]


@pytest.mark.asyncio
async def test_a_seed_that_cannot_be_told_is_reported_not_raised(db, matricxon_engine, monkeypatch):
    monkeypatch.setattr(ClusterService, "_push_to_seed", _REAL_PUSH)
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    cases = {
        400: ({"detail": "move the cache key first"}, "refused it: move the cache key first"),
        404: ({}, "has no cluster mode"),
        409: ({"detail": "this node is a joiner"}, "this node is a joiner"),
    }
    for status, (body, expected) in cases.items():
        await _first(db, A)
        state = await ClusterService(db, [A], _seed_replies(status, body)).set_enabled(True)
        assert state.enabled and expected in state.message

    def down(request):
        raise httpx.ConnectError("refused")

    await _first(db, A)
    state = await ClusterService(db, [A], httpx.MockTransport(down)).set_enabled(True)
    assert "could not be told" in state.message


@pytest.mark.asyncio
async def test_the_nodes_page_loads_the_renderers_in_order_and_has_its_controls(db, admin_user):
    html = (await pages.cluster_page(_request(admin_user.id), db=db)).body.decode()
    scripts = [
        "cluster_node_panel", "cluster_map_common", "cluster_map_layout", "cluster_map_graph",
        "cluster_map_dots", "cluster_map_dashboard", "cluster_map.js", "cluster.js",
    ]  # fmt: skip
    positions = [html.index(f"/static/js/{name}") for name in scripts]
    assert positions == sorted(positions)
    for element in ("cluster-map-search", "cluster-map-status", "cluster-map-motion", "cluster-map-canvas",
                    "cluster-map-dashboard", "cluster-map-tooltip", "cluster-map-tier"):  # fmt: skip
        assert f'id="{element}"' in html


def test_nothing_on_the_map_animates_unless_motion_is_on():
    """A continuously animated SVG took a laptop to 100% CPU. Every page animation must sit under.mx-motion and be
    stepped (low frame rate)."""
    import re
    from pathlib import Path

    html = (Path(__file__).parent.parent / "app" / "templates" / "cluster.html").read_text()
    css = html[html.index("<style>") : html.index("</style>")]
    for rule in re.findall(r"([^{}]+)\{([^{}]*animation:[^{}]*)\}", css):
        selector, body = rule
        if "@keyframes" in selector:
            continue
        assert ".mx-motion" in selector, f"{selector.strip()} animates all the time"
        assert "steps(" in body, f"{selector.strip()} is not stepped"
    js = Path(__file__).parent.parent / "app" / "static" / "js"
    for name in ("cluster_map_graph.js", "cluster_map_dots.js", "cluster_map.js"):
        text = (js / name).read_text()
        assert "setInterval" not in text or name == "cluster_map.js"
        assert "requestAnimationFrame" not in text or name == "cluster_map_dots.js"
    dots = (js / "cluster_map_dots.js").read_text()
    assert "requestAnimationFrame(() =>" in dots and "this.scheduled" in dots


def test_the_four_map_sizes_and_the_compact_poll_are_in_the_controller():
    from pathlib import Path

    controller = (Path(__file__).parent.parent / "app" / "static" / "js" / "cluster_map.js").read_text()
    for tier in (
        'name: "detail", max: 6',
        'name: "ring", max: 50',
        'name: "dots", max: 500',
        'name: "summary", max: Infinity',
    ):
        assert tier in controller
    assert "/api/cluster/map?compact=true" in controller  # big clusters never download every node's details
    assert "/api/cluster/map/node?id=" in controller
    assert "demo" in controller and "URLSearchParams" in controller


@pytest.mark.asyncio
async def test_the_images_warning_is_shown_with_the_image_settings_only_in_cluster_mode(db, monkeypatch):
    from app.routers import sdcpp_admin

    bad = ClusterCheck(
        id="images", status="action_needed", what="images: not on shared storage", why="w", hint="move it"
    )
    good = ClusterCheck(id="images", status="ok", what="images: shared and usable", why="w")
    verdict = {"check": bad}

    async def images_check(self):
        return verdict["check"]

    monkeypatch.setattr(ClusterService, "images_check", images_check)
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role = "off"
    await settings_service.set_matricxon_server_config(db, config)
    assert await sdcpp_admin.get_images_check(db, None) is None

    config.cluster_role = "seed"
    await settings_service.set_matricxon_server_config(db, config)
    assert (await sdcpp_admin.get_images_check(db, None)).hint == "move it"
    verdict["check"] = good
    assert await sdcpp_admin.get_images_check(db, None) is None


@pytest.mark.asyncio
async def test_cluster_mode_cannot_be_turned_off_while_other_nodes_exist(db, matricxon_engine, monkeypatch):
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role, config.cluster_hosts = "seed", ["http://10.0.0.7:8420"]
    await settings_service.set_matricxon_server_config(db, config)
    matricxon_pool.refresh_from_config(config)
    with pytest.raises(HTTPException) as refused:
        await cluster_admin.set_cluster(ClusterUpdate(enabled=False), db, None)
    assert refused.value.status_code == 409 and "Remove the other nodes" in refused.value.detail
    assert (await settings_service.get_matricxon_server_config(db)).cluster_role == "seed"

    config.cluster_hosts = []
    await settings_service.set_matricxon_server_config(db, config)
    matricxon_pool.refresh_from_config(config)
    assert (await cluster_admin.set_cluster(ClusterUpdate(enabled=False), db, None)).enabled is False


@pytest.mark.asyncio
async def test_in_remote_mode_pairing_writes_no_env_file_but_still_sets_a_remote_node_up(db, tmp_path, monkeypatch):
    """pAIring may manage no Matricxon: no .env write, but it can tell a remote node to become the seed."""
    written: list = []
    monkeypatch.setattr(matricxon_process.env_file, "write_env_values", lambda values, path: written.append(path))
    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda configured: tmp_path)
    config = await settings_service.get_matricxon_server_config(db)
    config.mode = "remote"
    matricxon_process.write_env_file(config)
    assert written == []
    config.mode = "local"
    matricxon_process.write_env_file(config)
    assert len(written) == 1

    from app.services.cluster_settings import ClusterSettings

    seen: list = []
    transport = _seed_replies(seen=seen)
    config.mode, config.cluster_role = "remote", "seed"
    note = await ClusterSettings(db, transport, [A, B]).push_to_seed(config, B)
    assert note is None and [s[1] for s in seen if s[2] == "/api/cluster/configure"] == ["b"]


@pytest.mark.asyncio
async def test_the_first_node_is_chosen_with_the_folders_and_gets_the_setup(db, matricxon_engine, monkeypatch):
    """No local Matricxon needed: the chosen remote host is told to start the cluster."""
    monkeypatch.setattr(ClusterService, "_push_to_seed", _REAL_PUSH)
    monkeypatch.setattr(matricxon_process, "write_env_file", lambda config: None)
    config = await settings_service.get_matricxon_server_config(db)
    config.mode, config.remote_hosts = "remote", [A, B]
    await settings_service.set_matricxon_server_config(db, config)
    matricxon_pool.refresh_from_config(config)
    seen: list = []
    service = ClusterService(db, [A, B], _seed_replies(seen=seen))

    with pytest.raises(ValueError, match="Choose the first node"):
        await service.set_enabled(True)
    stranger = await service.set_paths(ClusterPaths(first_node="stranger:8420"))
    assert stranger.paths.first_node == "http://stranger:8420"  # an unknown address becomes a host
    assert "http://stranger:8420" in matricxon_pool.get_effective_hosts()
    seen.clear()

    state = await service.set_paths(ClusterPaths(first_node=B, advertise_url=" http://10.0.0.9:8420/ "))
    assert state.paths.first_node == B
    assert [s[1] for s in seen if s[2] == "/api/cluster/configure"] == ["b"]
    seen.clear()
    await service.set_enabled(True)
    configure = [s for s in seen if s[2] == "/api/cluster/configure"]
    assert [s[1] for s in configure] == ["b"]
    assert json.loads(configure[0][3])["advertise_url"] == "http://10.0.0.9:8420"

    seen.clear()
    await service.set_paths(ClusterPaths(models_path="/m"))
    assert [s[1] for s in seen if s[2] == "/api/cluster/configure"] == ["b"]
    assert (await service.set_paths(ClusterPaths(first_node=""))).paths.first_node is None


@pytest.mark.asyncio
async def test_hosts_report_whether_they_are_fresh_only_before_consensus_runs(db, matricxon_engine):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/cluster/fresh":
            return httpx.Response(200, json={"fresh": False, "reasons": ["2 installed model(s)"]})
        if request.url.path == "/api/cluster/readiness":
            return httpx.Response(200, json=_readiness())
        return httpx.Response(200, json={"enabled": False})

    state = await ClusterService(db, [A], _transport(handler), with_fresh=True).state()
    assert state.hosts[0].fresh is False
    assert (await ClusterService(db, [A], _transport(handler)).state()).hosts[0].fresh is None


async def _save_folders(db, monkeypatch, role: str, roles: dict[str, str]) -> list[str]:
    """Hosts told to configure themselves when the folders are saved."""
    monkeypatch.setattr(ClusterService, "_push_to_seed", _REAL_PUSH)
    told: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/cluster/configure":
            told.append(request.url.host)
            return httpx.Response(200, json={"restarting": True})
        if request.url.path == "/api/cluster/readiness":
            return httpx.Response(200, json=_readiness(role=roles[request.url.host]))
        return httpx.Response(200, json={"enabled": False})

    await _first(db, B, hosts=(A, B))
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role = role
    await settings_service.set_matricxon_server_config(db, config)
    service = ClusterService(db, [A, B], _transport(handler))
    await service.set_paths(ClusterPaths(models_path="/mnt/m", prompt_cache_path="/mnt/c"))
    return told


@pytest.mark.asyncio
async def test_saving_the_folders_with_cluster_mode_off_tells_only_the_first_node(db, matricxon_engine, monkeypatch):
    assert await _save_folders(db, monkeypatch, "off", {"a": "off", "b": "off"}) == ["b"]


@pytest.mark.asyncio
async def test_saving_the_folders_before_consensus_tells_only_the_first_node(db, matricxon_engine, monkeypatch):
    assert await _save_folders(db, monkeypatch, "seed", {"a": "off", "b": "seed"}) == ["b"]
