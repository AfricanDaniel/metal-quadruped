"""Flask app: browse local/Jetson/sheep checkpoints, launch the MuJoCo viewer or export a policy, deploy/control the rea..."""

import datetime
import json
import os
import threading
import webbrowser

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for

from dashboard import (
    build_actions, download_actions, graphs, likes, local_actuator_actions, local_fs, local_tools_actions,
    policy_actions, procs, remote_fs, ros_actions, ssh, training_actions,
)
from dashboard.config import (
    JETSON_HOST, JETSON_WS_ROOT, MODELS_DIR, SHEEP_HOST, SHEEP_WS_ROOT,
)

app = Flask(__name__)
app.secret_key = 'dashboard-local-only'  # single-user local tool, not internet-facing


@app.template_filter('fmt_dt')
def fmt_dt(mtime):
    if not mtime:
        return '--'
    return datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')


# JSON polling endpoints (called every ~1.5s by page JS)
_POLL_ENDPOINTS = {
    'jetson_build_status', 'jetson_status_poll', 'sheep_download_status',
    'local_trainings_build_status', 'sheep_trainings_build_status',
}


_CONNECT_GATE_ENDPOINTS = {'jetson_root', 'sheep_root'}

# Fallback values shown in the "Actuator params" card before anything's ever been set/remembered
_ACTUATOR_PARAM_DEFAULTS = {'position_kp': '60', 'position_kd': '8', 'pose_speed_deg_s': '60'}


def _last_actuator_params():
    """Remembered actuator params, shared across the Basics AND Deploy pages."""

    remembered = procs.get_last_tool_form('actuator_params')
    return {name: remembered.get(name) or default
            for name, default in _ACTUATOR_PARAM_DEFAULTS.items()}


@app.before_request
def _remember_tab_position():
    """Lets the tab bar itself (base.html, via nav_url below) jump back to wherever you left off in a tab."""

    if request.method != 'GET' or request.endpoint in _POLL_ENDPOINTS:
        return
    if request.endpoint in _CONNECT_GATE_ENDPOINTS:
        return
    path = request.path
    endpoint = request.endpoint or ''
    if path.startswith('/local'):
        procs.set_last_path('local', path)
        # Sub-tab-specific tracking, same "forgets where I was" fix as jetson_policies below, for Local's own 4 sub-tabs
        if endpoint.startswith('local_trainings'):
            procs.set_last_path('local_trainings', path)
        elif endpoint.startswith('local_policy') or endpoint == 'local_policies':
            procs.set_last_path('local_policies', path)
        elif endpoint.startswith('local_tools'):
            procs.set_last_path('local_tools', path)
        else:
            procs.set_last_path('local_models', path)
    elif path.startswith('/jetson'):
        procs.set_last_path('jetson', path)
        if endpoint != 'jetson_basics':
            procs.set_last_path('jetson_policies', path)
    elif path.startswith('/sheep'):
        procs.set_last_path('sheep', path)
        if endpoint.startswith('sheep_trainings'):
            procs.set_last_path('sheep_trainings', path)
        else:
            procs.set_last_path('sheep_models', path)


@app.context_processor
def _inject_nav_urls():
    return {
        'nav_local_url': procs.get_last_path('local'),
        'nav_jetson_url': procs.get_last_path('jetson'),
        'nav_sheep_url': procs.get_last_path('sheep'),
        'nav_liked_url': url_for('liked_page'),
    }


@app.context_processor
def _inject_sub_tab_urls():
    """sub_tabs.html (Models/Trainings) is included from folders/files/ checkpoints/trainings templates on both Local and Sheep"""

    if request.path.startswith('/local'):
        return {
            'sub_models_url': procs.get_last_path('local_models'),
            'sub_trainings_url': procs.get_last_path('local_trainings'),
            'sub_policies_url': procs.get_last_path('local_policies'),
            'sub_tools_url': procs.get_last_path('local_tools'),
        }
    if request.path.startswith('/sheep'):
        return {
            'sub_models_url': procs.get_last_path('sheep_models'),
            'sub_trainings_url': procs.get_last_path('sheep_trainings'),
        }
    if request.path.startswith('/jetson'):
        return {
            'sub_jetson_policies_url': procs.get_last_path('jetson_policies'),
            'sub_jetson_basics_url': url_for('jetson_basics'),
            'sub_jetson_liked_url': url_for('jetson_liked'),
        }
    return {}


def _group_kwargs(group_name):
    """Jetson/local-policy routes take a policy group as TWO separate URL segments (<policies_dir>/<task>), not one slash-co..."""

    policies_dir, task = group_name.split('/', 1)
    return {'policies_dir': policies_dir, 'task': task}


def _suggest_redirect(label, url):
    """Queues a one-shot 'go to X?' popup for the NEXT page render."""

    session['redirect_suggestion'] = {'label': label, 'url': url}


@app.context_processor
def _inject_sheep_persistent_nav():
    """sheep_models_landing_url/sheep_trainings_landing_url: unlike sub_models_url/sub_trainings_url above (which follow pro..."""

    if not request.path.startswith('/sheep'):
        return {}
    return {
        'sheep_models_landing_url': url_for('sheep_home'),
        'sheep_trainings_landing_url': url_for('sheep_trainings'),
    }


@app.context_processor
def _inject_redirect_suggestion():
    return {'redirect_suggestion': session.pop('redirect_suggestion', None)}


def _suggest_action(label, url):
    """Queues a one-shot inline button rendered right in the flash banner on the NEXT page render."""

    session['action_suggestion'] = {'label': label, 'url': url}


@app.context_processor
def _inject_action_suggestion():
    return {'action_suggestion': session.pop('action_suggestion', None)}


# ====================================================================== Local ======================================================================

@app.route('/')
def index():
    return redirect(url_for('local_folders'))


@app.route('/local')
def local_folders():
    folders = [
        {'name': f['name'], 'mtime': f['mtime'], 'url': url_for('local_files', folder=f['name'])}
        for f in local_fs.list_model_folders()
    ]
    return render_template(
        'folders.html', tab='local', active_tab='local', sub_tab='models',
        heading='Local models/', subtitle=MODELS_DIR, folders=folders,
    )


@app.route('/local/<folder>')
def local_files(folder):
    fnames = [
        {**f, 'url': url_for('local_checkpoints', folder=folder, fname=f['fname']),
         'graphs_url': url_for('local_trainings_graphs', fname=f['fname'])}
        for f in local_fs.list_fnames(folder)
    ]
    sidebar_items = [
        {'label': f['name'], 'active': f['name'] == folder, 'url': url_for('local_files', folder=f['name'])}
        for f in local_fs.list_model_folders()
    ]
    return render_template(
        'files.html', tab='local', active_tab='local', sub_tab='models',
        folder=folder, fnames=fnames, back_url=url_for('local_folders'), sidebar_items=sidebar_items,
    )


@app.route('/local/<folder>/<fname>')
def local_checkpoints(folder, fname):
    checkpoints = [
        {**c, 'url': url_for('local_detail', folder=folder, fname=fname, basename=c['basename']),
         'view_url': url_for('local_view_training', folder=folder, fname=fname, basename=c['basename']),
         'delete_url': url_for('local_delete_checkpoint', folder=folder, fname=fname, basename=c['basename']),
         'like_url': url_for('local_toggle_like', folder=folder, fname=fname, basename=c['basename']),
         'liked': likes.is_liked('local', folder, fname, c['basename']),
         'last_start_pose': procs.get_last_tool_form(f"view_training:{c['basename']}").get('start_pose', 'home'),
         'last_max_slew_deg_per_s': procs.get_last_tool_form(f"view_training:{c['basename']}").get('max_slew_deg_per_s', ''),
         'last_joint_stiffness': procs.get_last_tool_form(f"view_training:{c['basename']}").get('joint_stiffness', '')}
        for c in local_fs.list_checkpoints(folder, fname)
    ]
    # back_url: this page has TWO real entry points
    back_url = (url_for('local_trainings_graphs', fname=fname) if request.args.get('from') == 'graphs'
                else url_for('local_files', folder=folder))
    return render_template(
        'checkpoints.html', tab='local', active_tab='local', sub_tab='models',
        folder=folder, fname=fname, checkpoints=checkpoints,
        env_id=local_fs.resolve_env_id(folder),
        back_url=back_url,
        graphs_url=url_for('local_trainings_graphs', fname=fname),
        tabs_graphs_url=url_for('local_trainings_graphs', fname=fname, **{'from': 'checkpoints'}),
        tabs_checkpoints_url=url_for('local_checkpoints', folder=folder, fname=fname),
        active_training_tab='checkpoints',
    )


@app.route('/local/<folder>/<fname>/<basename>')
def local_detail(folder, fname, basename):
    checkpoints = local_fs.list_checkpoints(folder, fname)
    checkpoint = next((c for c in checkpoints if c['basename'] == basename), None)
    if checkpoint is None:
        checkpoint = {'basename': basename, 'mtime': 0, 'steps': None}
    sidebar_items = [
        {'label': c['basename'], 'active': c['basename'] == basename,
         'url': url_for('local_detail', folder=folder, fname=fname, basename=c['basename'])}
        for c in checkpoints
    ]
    return render_template(
        'local_detail.html', tab='local', active_tab='local',
        folder=folder, fname=fname, checkpoint=checkpoint, sidebar_items=sidebar_items,
        resolved_env_id=local_fs.resolve_env_id(folder),
        action_url=url_for('local_detail_act', folder=folder, fname=fname, basename=basename),
        delete_url=url_for('local_delete_checkpoint', folder=folder, fname=fname, basename=basename),
        back_url=url_for('local_checkpoints', folder=folder, fname=fname),
    )


@app.route('/local/<folder>/<fname>/<basename>/delete', methods=['POST'])
def local_delete_checkpoint(folder, fname, basename):
    ok = local_fs.delete_local_file(local_fs.checkpoint_zip_path(folder, basename))
    flash(f'Deleted {basename}.' if ok else f'Failed to delete {basename}.', 'success' if ok else 'error')
    return redirect(url_for('local_checkpoints', folder=folder, fname=fname))


def _do_view_training(host, folder, fname, basename, redirect_url):
    """Shared by local_view_training()/sheep_view_training()/ liked_view_training()."""

    if host == 'sheep':
        local_path = local_fs.checkpoint_zip_path(folder, basename)
        if not os.path.exists(local_path):
            ok, local_path = _sheep_download(folder, basename)
            if not ok:
                flash('Download failed, cannot view training.', 'error')
                return redirect(redirect_url)
    else:
        local_path = local_fs.checkpoint_zip_path(folder, basename)
    env_id = local_fs.resolve_env_id(folder)
    start_pose = request.form.get('start_pose', 'home')
    max_slew_deg_per_s = request.form.get('max_slew_deg_per_s', '').strip()
    joint_stiffness = request.form.get('joint_stiffness', '').strip()
    if env_id == 'Dog-Walk-v0':
        procs.set_last_tool_form(f'view_training:{basename}', {
            'start_pose': start_pose, 'max_slew_deg_per_s': max_slew_deg_per_s,
            'joint_stiffness': joint_stiffness})
    ok, info = policy_actions.launch_view_training(
        local_path, env_id, episodes=5,
        start_pose=start_pose if env_id == 'Dog-Walk-v0' else None,
        control_mode='position',
        max_slew_deg_per_s=max_slew_deg_per_s or policy_actions.VIEW_DEFAULT_MAX_SLEW_DEG_PER_S,
        joint_stiffness=joint_stiffness or None,
    )
    flash(('Downloaded and launched MuJoCo viewer.' if host == 'sheep' else 'Launched MuJoCo viewer.') if ok
          else f'Failed to launch: {info}', 'success' if ok else 'error')
    return redirect(redirect_url)


@app.route('/local/<folder>/<fname>/<basename>/view', methods=['POST'])
def local_view_training(folder, fname, basename):
    """One-click viewer launch straight from the checkpoints list, so seeing a checkpoint run doesn't require clicking into ..."""

    return _do_view_training('local', folder, fname, basename,
                              url_for('local_checkpoints', folder=folder, fname=fname))


@app.route('/local/<folder>/<fname>/<basename>/like', methods=['POST'])
def local_toggle_like(folder, fname, basename):
    likes.toggle_like('local', folder, fname, basename)
    return redirect(url_for('local_checkpoints', folder=folder, fname=fname))


@app.route('/local/<folder>/<fname>/<basename>/act', methods=['POST'])
def local_detail_act(folder, fname, basename):
    env_id = request.form.get('env_id', 'Dog-Walk-v0')
    control_mode = request.form.get('control_mode', 'position')
    action = request.form.get('action')
    zip_path = local_fs.checkpoint_zip_path(folder, basename)

    if action == 'view':
        episodes = request.form.get('episodes', '5')
        start_pose = request.form.get('start_pose', 'home')
        ok, info = policy_actions.launch_view_training(
            zip_path, env_id, episodes,
            start_pose=start_pose if env_id == 'Dog-Walk-v0' else None,
            control_mode=control_mode,
        )
        flash('Launched MuJoCo viewer.' if ok else f'Failed to launch: {info}',
              'success' if ok else 'error')
    elif action == 'export':
        ok, message, _ = policy_actions.run_export_policy(zip_path, env_id, control_mode, basename, folder)
        flash(message, 'success' if ok else 'error')
        if ok:
            policies_dir, task = policy_actions.export_target_group(env_id, control_mode, folder)
            _suggest_redirect('Go to policy', url_for(
                'local_policy_detail', policies_dir=policies_dir, task=task, fname=fname, basename=basename))
    else:
        flash(f'Unknown action: {action}', 'error')

    return redirect(url_for('local_detail', folder=folder, fname=fname, basename=basename))


@app.route('/local/trainings')
def local_trainings():
    running = [
        {**r, 'graphs_url': url_for('local_trainings_graphs', fname=r['fname']),
         'stop_url': url_for('local_trainings_stop', fname=r['fname']),
         'checkpoint_count': local_fs.count_checkpoints_for_fname(r['fname'])}
        for r in training_actions.list_running_local_trainings()
    ]
    model_folders = [f['name'] for f in local_fs.list_model_folders()]
    return render_template(
        'trainings.html', tab='local', active_tab='local', sub_tab='trainings',
        running=running, model_folders=model_folders,
        launch_url=url_for('local_trainings_launch'),
        build=build_actions.local_build_status(),
        build_url=url_for('local_trainings_build'),
        build_status_url=url_for('local_trainings_build_status'),
    )


@app.route('/local/trainings/launch', methods=['POST'])
def local_trainings_launch():
    ok, message = training_actions.launch_local_training(request.form)
    flash(message, 'success' if ok else 'error')
    return redirect(url_for('local_trainings'))


@app.route('/local/trainings/build', methods=['POST'])
def local_trainings_build():
    ok, message = build_actions.start_local_build()
    flash(message, 'success' if ok else 'error')
    return redirect(url_for('local_trainings'))


@app.route('/local/trainings/build_status')
def local_trainings_build_status():
    return jsonify(build_actions.local_build_status())


@app.route('/local/trainings/<fname>/stop', methods=['POST'])
def local_trainings_stop(fname):
    ok = training_actions.stop_training(fname)
    flash(f'Stopped "{fname}".' if ok else f'Failed to stop "{fname}" -- check it directly.',
          'success' if ok else 'error')
    return redirect(url_for('local_trainings'))


@app.route('/local/trainings/<fname>/graphs')
def local_trainings_graphs(fname):
    scalars = graphs.read_scalars(fname)
    folder = local_fs.find_folder_for_fname(fname)
    # back_url: mirrors local_checkpoints()'s own ?from=graphs marker
    back_url = (url_for('local_checkpoints', folder=folder, fname=fname)
                if request.args.get('from') == 'checkpoints' and folder
                else url_for('local_trainings'))
    return render_template(
        'graphs.html', tab='local', active_tab='local',
        fname=fname, running=training_actions.is_training_running(fname), sheep=False,
        request_path=request.path, tags=graphs.ordered_tags(scalars), series_json=json.dumps(scalars),
        back_url=back_url,
        tabs_graphs_url=request.path,
        tabs_checkpoints_url=(url_for('local_checkpoints', folder=folder, fname=fname, **{'from': 'graphs'})
                               if folder else None),
        active_training_tab='graphs',
    )


@app.route('/local/policies')
def local_policies():
    groups = [
        {'name': g['name'], 'mtime': g['mtime'], 'url': url_for('local_policy_files', **_group_kwargs(g['name']))}
        for g in local_fs.list_policy_groups()
    ]
    return render_template(
        'folders.html', tab='local', active_tab='local', sub_tab='policies',
        heading='Local policies', subtitle='src/policies_position, src/policies_torque', folders=groups,
    )


@app.route('/local/policies/<policies_dir>/<task>')
def local_policy_files(policies_dir, task):
    group_name = f'{policies_dir}/{task}'
    jetson_set = remote_fs.basenames_on_jetson(JETSON_HOST, group_name)
    fnames = []
    for f in local_fs.list_policy_fnames(group_name):
        checkpoints = local_fs.list_policy_checkpoints(group_name, f['fname'])
        jetson_count = sum(1 for c in checkpoints if c['basename'] in jetson_set)
        fnames.append({
            **f, 'jetson_count': jetson_count,
            'url': url_for('local_policy_checkpoints', policies_dir=policies_dir, task=task, fname=f['fname']),
        })
    sidebar_items = [
        {'label': g['name'], 'active': g['name'] == group_name, 'url': url_for('local_policy_files', **_group_kwargs(g['name']))}
        for g in local_fs.list_policy_groups()
    ]
    return render_template(
        'policy_files.html', tab='local', active_tab='local', sub_tab='policies',
        group_name=group_name, fnames=fnames, back_url=url_for('local_policies'), sidebar_items=sidebar_items,
    )


@app.route('/local/policies/<policies_dir>/<task>/<fname>')
def local_policy_checkpoints(policies_dir, task, fname):
    group_name = f'{policies_dir}/{task}'
    jetson_set = remote_fs.basenames_on_jetson(JETSON_HOST, group_name)
    checkpoints = [
        {**c, 'on_jetson': c['basename'] in jetson_set,
         'url': url_for('local_policy_detail', policies_dir=policies_dir, task=task,
                         fname=fname, basename=c['basename']),
         'delete_url': url_for('local_policy_delete', policies_dir=policies_dir, task=task,
                                fname=fname, basename=c['basename'])}
        for c in local_fs.list_policy_checkpoints(group_name, fname)
    ]
    return render_template(
        'policy_checkpoints.html', tab='local', active_tab='local', sub_tab='policies',
        group_name=group_name, fname=fname, checkpoints=checkpoints,
        back_url=url_for('local_policy_files', policies_dir=policies_dir, task=task),
    )


@app.route('/local/policies/<policies_dir>/<task>/<fname>/<basename>')
def local_policy_detail(policies_dir, task, fname, basename):
    group_name = f'{policies_dir}/{task}'
    checkpoints = local_fs.list_policy_checkpoints(group_name, fname)
    checkpoint = next((c for c in checkpoints if c['basename'] == basename), {'basename': basename, 'mtime': 0})
    csvs = [
        {**c, 'delete_url': url_for('local_policy_delete_csv', policies_dir=policies_dir, task=task,
                                     fname=fname, basename=basename, csv_name=c['name'])}
        for c in local_fs.list_policy_csvs(group_name, basename)
    ]
    on_jetson = basename in remote_fs.basenames_on_jetson(JETSON_HOST, group_name)
    sidebar_items = [
        {'label': c['basename'], 'active': c['basename'] == basename,
         'url': url_for('local_policy_detail', policies_dir=policies_dir, task=task,
                         fname=fname, basename=c['basename'])}
        for c in checkpoints
    ]
    return render_template(
        'policy_detail.html', tab='local', active_tab='local',
        group_name=group_name, fname=fname, checkpoint=checkpoint, csvs=csvs, on_jetson=on_jetson,
        sidebar_items=sidebar_items,
        back_url=url_for('local_policy_checkpoints', policies_dir=policies_dir, task=task, fname=fname),
        upload_url=url_for('local_policy_upload', policies_dir=policies_dir, task=task,
                            fname=fname, basename=basename),
        delete_url=url_for('local_policy_delete', policies_dir=policies_dir, task=task,
                            fname=fname, basename=basename),
    )


@app.route('/local/policies/<policies_dir>/<task>/<fname>/<basename>/upload', methods=['POST'])
def local_policy_upload(policies_dir, task, fname, basename):
    group_name = f'{policies_dir}/{task}'
    local_path = local_fs.policy_pt_path(group_name, basename)
    remote_path = remote_fs.jetson_policy_pt_path(group_name, basename)
    result = ssh.scp_upload(JETSON_HOST, local_path, remote_path)
    flash('Uploaded to jetson.' if result.ok else f'Upload failed: {(result.stderr or result.stdout).strip()[:200]}',
          'success' if result.ok else 'error')
    if result.ok:
        _suggest_redirect('Go to jetson', url_for(
            'jetson_detail', policies_dir=policies_dir, task=task, fname=fname, basename=basename))
    return redirect(url_for('local_policy_detail', policies_dir=policies_dir, task=task,
                             fname=fname, basename=basename))


@app.route('/local/policies/<policies_dir>/<task>/<fname>/<basename>/delete', methods=['POST'])
def local_policy_delete(policies_dir, task, fname, basename):
    group_name = f'{policies_dir}/{task}'
    # Delete associated CSVs before the .pt, so an interruption between
    # the two deletes never leaves an orphaned CSV behind.
    csv_names = [c['name'] for c in local_fs.list_policy_csvs(group_name, basename)]
    csvs_ok = all(local_fs.delete_local_file(local_fs.policy_csv_path(group_name, name)) for name in csv_names)
    ok = local_fs.delete_local_file(local_fs.policy_pt_path(group_name, basename)) and csvs_ok
    if csv_names:
        flash(f'Deleted {basename} and {len(csv_names)} associated CSV(s).' if ok else f'Failed to delete {basename}.',
              'success' if ok else 'error')
    else:
        flash(f'Deleted {basename}.' if ok else f'Failed to delete {basename}.', 'success' if ok else 'error')
    return redirect(url_for('local_policy_checkpoints', policies_dir=policies_dir, task=task, fname=fname))


@app.route('/local/policies/<policies_dir>/<task>/<fname>/<basename>/csv/<csv_name>/delete', methods=['POST'])
def local_policy_delete_csv(policies_dir, task, fname, basename, csv_name):
    group_name = f'{policies_dir}/{task}'
    ok = local_fs.delete_local_file(local_fs.policy_csv_path(group_name, csv_name))
    flash(f'Deleted {csv_name}.' if ok else f'Failed to delete {csv_name}.', 'success' if ok else 'error')
    return redirect(url_for('local_policy_detail', policies_dir=policies_dir, task=task,
                             fname=fname, basename=basename))


@app.route('/local/tools')
def local_tools():
    return render_template(
        'tools.html', tab='local', active_tab='local', sub_tab='tools',
        mjcf_choices=local_tools_actions.MJCF_CHOICES,
        last_save_pose_form=procs.get_last_tool_form('save_pose'),
        last_mmc_form=procs.get_last_tool_form('manual_motor_control'),
        last_vbd_form=procs.get_last_tool_form('verify_belt_decoupling'),
        actuator_status=local_actuator_actions.local_actuator_status(),
        actuator_toggle_url=url_for('local_actuator_toggle'),
        read_motor_url=url_for('local_read_motor'),
        adjust_motor_url=url_for('local_adjust_motor'),
    )


@app.route('/local/tools/actuator/toggle', methods=['POST'])
def local_actuator_toggle():
    status = local_actuator_actions.local_actuator_status()
    if status['running']:
        ok = local_actuator_actions.stop_local_actuator()
        flash('Stopped basic_control.' if ok else 'Failed to stop basic_control -- check directly.',
              'success' if ok else 'error')
    else:
        ok, message = local_actuator_actions.start_local_actuator()
        flash(message, 'success' if ok else 'error')
    return redirect(url_for('local_tools'))


@app.route('/local/tools/read_motor', methods=['POST'])
def local_read_motor():
    raw = request.form.get('motor_ids', '').strip()
    motor_ids = [int(x) for x in raw.split(',') if x.strip().isdigit()] if raw else None
    result = ros_actions.local_read_motor_positions(motor_ids)
    flash(result.stdout.strip() or result.stderr.strip() or 'read_motor_positions called.',
          'success' if result.ok else 'error')
    return redirect(url_for('local_tools'))


@app.route('/local/tools/adjust_motor', methods=['POST'])
def local_adjust_motor():
    motor_id = request.form.get('motor_id')
    degrees = request.form.get('degrees')
    if not motor_id or not degrees:
        flash('motor_id and degrees are both required.', 'error')
        return redirect(url_for('local_tools'))
    result = ros_actions.local_adjust_motor_position(motor_id, degrees)
    flash(result.stdout.strip() or result.stderr.strip() or 'adjust_motor_position called.',
          'success' if result.ok else 'error')
    return redirect(url_for('local_tools'))


@app.route('/local/tools/save_pose', methods=['POST'])
def local_tools_save_pose():
    procs.set_last_tool_form('save_pose', request.form)
    ok, info = local_tools_actions.launch_save_pose(request.form)
    flash('Launched save_pose.py.' if ok else f'Failed to launch: {info}', 'success' if ok else 'error')
    return redirect(url_for('local_tools'))


@app.route('/local/tools/manual_motor_control', methods=['POST'])
def local_tools_manual_motor_control():
    procs.set_last_tool_form('manual_motor_control', request.form)
    ok, info = local_tools_actions.launch_manual_motor_control(request.form)
    flash('Launched manual_motor_control.' if ok else f'Failed to launch: {info}', 'success' if ok else 'error')
    return redirect(url_for('local_tools'))


@app.route('/local/tools/verify_belt_decoupling', methods=['POST'])
def local_tools_verify_belt_decoupling():
    procs.set_last_tool_form('verify_belt_decoupling', request.form)
    ok, info = local_tools_actions.launch_verify_belt_decoupling(request.form)
    flash('Launched verify_belt_decoupling.' if ok else f'Failed to launch: {info}', 'success' if ok else 'error')
    return redirect(url_for('local_tools'))


@app.route('/local/tools/dog_view', methods=['POST'])
def local_tools_dog_view():
    ok, info = local_tools_actions.launch_dog_view()
    flash('Launched dog_view.launch.py.' if ok else f'Failed to launch: {info}', 'success' if ok else 'error')
    return redirect(url_for('local_tools'))


# ====================================================================== Jetson ======================================================================

@app.route('/jetson')
def jetson_root():
    if procs.is_connected('jetson'):
        return redirect(url_for('jetson_home'))
    return render_template('jetson_connect.html', active_tab='jetson',
                            connect_url=url_for('jetson_connect'), failed=False)


@app.route('/jetson/connect', methods=['POST'])
def jetson_connect():
    ok = ssh.test_connection(JETSON_HOST)
    procs.set_connected('jetson', ok)
    if ok:
        return redirect(url_for('jetson_home'))
    return render_template('jetson_connect.html', active_tab='jetson',
                            connect_url=url_for('jetson_connect'), failed=True)


def _require_jetson():
    if not procs.is_connected('jetson'):
        return redirect(url_for('jetson_root'))
    return None


@app.route('/jetson/home')
def jetson_home():
    guard = _require_jetson()
    if guard:
        return guard
    groups = [
        {'name': g['name'], 'mtime': g['mtime'], 'url': url_for('jetson_group_files', **_group_kwargs(g['name']))}
        for g in remote_fs.list_jetson_groups(JETSON_HOST)
    ]
    return render_template(
        'jetson_home.html', tab='jetson', active_tab='jetson',
        heading='Jetson policies', subtitle=f'{JETSON_HOST}:{JETSON_WS_ROOT}',
        folders=groups,
        build=procs.build_status(),
        build_url=url_for('jetson_build', next=request.path),
        build_status_url=url_for('jetson_build_status'),
    )


@app.route('/jetson/basics')
def jetson_basics():
    guard = _require_jetson()
    if guard:
        return guard
    return render_template(
        'jetson_basics.html', tab='jetson', active_tab='jetson', sub_tab='basics',
        JETSON_HOST=JETSON_HOST, JETSON_WS_ROOT=JETSON_WS_ROOT,
        hw_status=procs.hardware_bringup_status(),
        hw_toggle_url=url_for('jetson_hw_toggle', next=request.path),
        status_poll_url=url_for('jetson_status_poll'),
        set_home_url=url_for('jetson_set_home', next=request.path),
        go_to_pose_url=url_for('jetson_go_to_pose', next=request.path),
        last_pose_speed_deg_s=procs.get_last_tool_form('go_to_pose').get('speed_deg_s') or '60',
        last_pose_log_csv=procs.get_last_tool_form('go_to_pose').get('log_csv'),
        read_motor_url=url_for('jetson_read_motor', next=request.path),
        adjust_motor_url=url_for('jetson_adjust_motor', next=request.path),
        reset_motors_url=url_for('jetson_reset_motors', next=request.path),
        set_actuator_params_url=url_for('jetson_set_actuator_params', next=request.path),
        last_actuator_params=_last_actuator_params(),
        last_service_output=request.args.get('out'),
        build=procs.build_status(),
        build_url=url_for('jetson_build', next=request.path),
        build_status_url=url_for('jetson_build_status'),
    )


@app.route('/jetson/build', methods=['POST'])
def jetson_build():
    ok = procs.start_build()
    flash('Build started.' if ok else 'Failed to start build.', 'success' if ok else 'error')
    return _redirect_next('jetson_home')


@app.route('/jetson/build_status')
def jetson_build_status():
    return jsonify(procs.build_status())


@app.route('/jetson/<policies_dir>/<task>')
def jetson_group_files(policies_dir, task):
    guard = _require_jetson()
    if guard:
        return guard
    group_name = f'{policies_dir}/{task}'
    fnames = [
        {**f, 'url': url_for('jetson_group_checkpoints', policies_dir=policies_dir, task=task, fname=f['fname'])}
        for f in remote_fs.list_jetson_fnames(JETSON_HOST, group_name)
    ]
    sidebar_items = [
        {'label': g['name'], 'active': g['name'] == group_name, 'url': url_for('jetson_group_files', **_group_kwargs(g['name']))}
        for g in remote_fs.list_jetson_groups(JETSON_HOST)
    ]
    return render_template(
        'files.html', tab='jetson', active_tab='jetson',
        folder=group_name, fnames=fnames, back_url=url_for('jetson_home'), sidebar_items=sidebar_items,
    )


@app.route('/jetson/<policies_dir>/<task>/<fname>')
def jetson_group_checkpoints(policies_dir, task, fname):
    guard = _require_jetson()
    if guard:
        return guard
    group_name = f'{policies_dir}/{task}'
    checkpoints = [
        {**c, 'url': url_for('jetson_detail', policies_dir=policies_dir, task=task,
                              fname=fname, basename=c['basename']),
         'delete_url': url_for('jetson_delete_policy', policies_dir=policies_dir, task=task,
                                fname=fname, basename=c['basename']),
         'like_url': url_for('jetson_toggle_like', policies_dir=policies_dir, task=task,
                              fname=fname, basename=c['basename']),
         'liked': likes.is_liked('jetson', group_name, fname, c['basename'])}
        for c in remote_fs.list_jetson_checkpoints(JETSON_HOST, group_name, fname)
    ]
    return render_template(
        'checkpoints.html', tab='jetson', active_tab='jetson',
        folder=group_name, fname=fname, checkpoints=checkpoints,
        back_url=url_for('jetson_group_files', policies_dir=policies_dir, task=task),
    )


@app.route('/jetson/<policies_dir>/<task>/<fname>/<basename>/like', methods=['POST'])
def jetson_toggle_like(policies_dir, task, fname, basename):
    likes.toggle_like('jetson', f'{policies_dir}/{task}', fname, basename)
    return redirect(url_for('jetson_group_checkpoints', policies_dir=policies_dir, task=task, fname=fname))


@app.route('/jetson/liked')
def jetson_liked():
    guard = _require_jetson()
    if guard:
        return guard
    items = []
    for item in likes.list_likes():
        if item['host'] != 'jetson':
            continue
        policies_dir, task = item['folder'].split('/', 1)
        items.append({
            **item,
            'policies_dir': policies_dir,
            'task': task,
            'detail_url': url_for('jetson_detail', policies_dir=policies_dir, task=task,
                                   fname=item['fname'], basename=item['basename']),
            'unlike_url': url_for('jetson_toggle_like', policies_dir=policies_dir, task=task,
                                   fname=item['fname'], basename=item['basename']),
        })
    return render_template('jetson_liked.html', tab='jetson', active_tab='jetson', sub_tab='liked', items=items)


@app.route('/jetson/<policies_dir>/<task>/<fname>/<basename>')
def jetson_detail(policies_dir, task, fname, basename):
    guard = _require_jetson()
    if guard:
        return guard
    group_name = f'{policies_dir}/{task}'
    checkpoints = remote_fs.list_jetson_checkpoints(JETSON_HOST, group_name, fname)
    checkpoint = next((c for c in checkpoints if c['basename'] == basename), {'basename': basename, 'mtime': 0})
    csvs = [
        {**c, 'local': os.path.exists(local_fs.policy_csv_path(group_name, c['name'])),
         'download_url': url_for('jetson_csv_download', policies_dir=policies_dir, task=task, fname=fname,
                                  basename=basename, csv_name=c['name']),
         'delete_url': url_for('jetson_csv_delete', policies_dir=policies_dir, task=task, fname=fname,
                                basename=basename, csv_name=c['name'])}
        for c in remote_fs.list_jetson_csvs(JETSON_HOST, group_name, basename)
    ]
    dstatus = procs.deploy_status()
    sidebar_items = [
        {'label': c['basename'], 'active': c['basename'] == basename,
         'url': url_for('jetson_detail', policies_dir=policies_dir, task=task, fname=fname, basename=c['basename'])}
        for c in checkpoints
    ]
    return render_template(
        'jetson_deploy_detail.html', tab='jetson', active_tab='jetson',
        group_name=group_name, fname=fname, checkpoint=checkpoint, csvs=csvs, sidebar_items=sidebar_items,
        back_url=url_for('jetson_group_checkpoints', policies_dir=policies_dir, task=task, fname=fname),
        deploy_status=dstatus,
        last_deploy_form=procs.get_last_deploy_form(),
        deploy_toggle_url=url_for('jetson_deploy_toggle', policies_dir=policies_dir, task=task,
                                   fname=fname, basename=basename),
        status_poll_url=url_for('jetson_status_poll'),
        go_to_pose_url=url_for('jetson_go_to_pose', next=request.path),
        set_actuator_params_url=url_for('jetson_set_actuator_params', next=request.path),
        last_actuator_params=_last_actuator_params(),
        delete_url=url_for('jetson_delete_policy', policies_dir=policies_dir, task=task,
                            fname=fname, basename=basename),
    )


@app.route('/jetson/<policies_dir>/<task>/<fname>/<basename>/csv/<csv_name>/download', methods=['POST'])
def jetson_csv_download(policies_dir, task, fname, basename, csv_name):
    guard = _require_jetson()
    if guard:
        return guard
    group_name = f'{policies_dir}/{task}'
    remote_path = remote_fs.jetson_csv_path(group_name, csv_name)
    local_path = local_fs.policy_csv_path(group_name, csv_name)
    result = ssh.scp_download(JETSON_HOST, remote_path, local_path)
    flash(f'Downloaded {csv_name}.' if result.ok else 'Download failed.', 'success' if result.ok else 'error')
    return redirect(url_for('jetson_detail', policies_dir=policies_dir, task=task, fname=fname, basename=basename))


@app.route('/jetson/<policies_dir>/<task>/<fname>/<basename>/csv/<csv_name>/delete', methods=['POST'])
def jetson_csv_delete(policies_dir, task, fname, basename, csv_name):
    guard = _require_jetson()
    if guard:
        return guard
    group_name = f'{policies_dir}/{task}'
    ok = ssh.remove_remote_file(JETSON_HOST, remote_fs.jetson_csv_path(group_name, csv_name))
    flash(f'Deleted {csv_name}.' if ok else f'Failed to delete {csv_name}.', 'success' if ok else 'error')
    return redirect(url_for('jetson_detail', policies_dir=policies_dir, task=task, fname=fname, basename=basename))


@app.route('/jetson/<policies_dir>/<task>/<fname>/<basename>/delete', methods=['POST'])
def jetson_delete_policy(policies_dir, task, fname, basename):
    guard = _require_jetson()
    if guard:
        return guard
    group_name = f'{policies_dir}/{task}'
    csv_names = [c['name'] for c in remote_fs.list_jetson_csvs(JETSON_HOST, group_name, basename)]
    csvs_ok = all(ssh.remove_remote_file(JETSON_HOST, remote_fs.jetson_csv_path(group_name, name))
                  for name in csv_names)
    ok = ssh.remove_remote_file(JETSON_HOST, remote_fs.jetson_policy_pt_path(group_name, basename)) and csvs_ok
    if csv_names:
        flash(f'Deleted {basename} and {len(csv_names)} associated CSV(s) from jetson.' if ok
              else f'Failed to delete {basename}.', 'success' if ok else 'error')
    else:
        flash(f'Deleted {basename} from jetson.' if ok else f'Failed to delete {basename}.', 'success' if ok else 'error')
    return redirect(url_for('jetson_group_checkpoints', policies_dir=policies_dir, task=task, fname=fname))


def _redirect_next(default_endpoint):
    nxt = request.args.get('next') or request.form.get('next')
    return redirect(nxt) if nxt else redirect(url_for(default_endpoint))


@app.route('/jetson/hw_toggle', methods=['POST'])
def jetson_hw_toggle():
    status = procs.hardware_bringup_status()
    if status['running']:
        ok = procs.stop_hardware_bringup()
        flash('Hardware bringup stopped.' if ok else 'Failed to stop hardware bringup.',
              'success' if ok else 'error')
    else:
        ok, message = procs.start_hardware_bringup()
        flash(message, 'success' if ok else 'error')
    return _redirect_next('jetson_home')


def _expand_tilde_for_ros_arg(path):
    """A '-p key:=value' ROS arg is just another bash word in the exec'd command line, but bash only tilde-expands a '~' at ..."""

    if path.startswith('~/'):
        return '$HOME/' + path[2:]
    return path


def _ros_float(value, default):
    """Formats a float-typed ROS parameter override so it's ALWAYS unambiguously a float on the wire, e.g."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@app.route('/jetson/<policies_dir>/<task>/<fname>/<basename>/deploy_toggle', methods=['POST'])
def jetson_deploy_toggle(policies_dir, task, fname, basename):
    group_name = f'{policies_dir}/{task}'
    status = procs.deploy_status()
    if status['running']:
        ok = procs.stop_deploy()
        flash('Deployment stopped.' if ok else 'Failed to stop deployment.', 'success' if ok else 'error')
    else:
        procs.set_last_deploy_form(request.form)
        policy_pt = remote_fs.jetson_policy_pt_path(group_name, basename)
        ros_args = [f'-p policy_path:={_expand_tilde_for_ros_arg(policy_pt)}']
        ros_args.append(f'-p dry_run_hold_pose:={"true" if request.form.get("dry_run_hold_pose") else "false"}')
        if request.form.get('max_delta_deg_per_step_enabled'):
            ros_args.append(f'-p max_delta_deg_per_step:={_ros_float(request.form.get("max_delta_deg_per_step"), 2.0)}')
        if request.form.get('max_target_lead_deg_enabled'):
            ros_args.append(f'-p max_target_lead_deg:={_ros_float(request.form.get("max_target_lead_deg"), 10.0)}')
        if request.form.get('freeze_after_sec_enabled'):
            ros_args.append(f'-p freeze_after_sec:={_ros_float(request.form.get("freeze_after_sec"), 5.0)}')
        if request.form.get('control_rate_hz_enabled'):
            ros_args.append(f'-p control_rate_hz:={_ros_float(request.form.get("control_rate_hz"), 100.0)}')
        if request.form.get('log_csv_enabled'):
            log_csv_name = request.form.get('log_csv') or basename
            if not log_csv_name.endswith('.csv'):
                log_csv_name += '.csv'
            log_csv_path = remote_fs.jetson_csv_path(group_name, log_csv_name)
            ros_args.append(f'-p log_csv:={_expand_tilde_for_ros_arg(log_csv_path)}')
        if request.form.get('home_position_deg_cache_path_enabled'):
            cache_path = request.form.get('home_position_deg_cache_path') or '~/.dog_home_cache.yaml'
            ros_args.append(f'-p home_position_deg_cache_path:={_expand_tilde_for_ros_arg(cache_path)}')
        if request.form.get('home_reference_mode') == 'edited':
            back_fraction = _ros_float(request.form.get('home_switch_back_leg_fraction'), 1.0)
            ros_args.append(f'-p home_switch_back_leg_fraction:={back_fraction}')
            # front_fraction (see home_correction.py's FRONT_LEG_HOME_ CORRECTION_DEG) defaults to 0.0 (inert) unless the form field is explicitly non-blank
            front_fraction_raw = request.form.get('home_switch_front_leg_fraction', '').strip()
            if front_fraction_raw:
                ros_args.append(f'-p home_switch_front_leg_fraction:={_ros_float(front_fraction_raw, 0.0)}')
            ros_args.append(f'-p home_switch_after_sec:={_ros_float(request.form.get("home_switch_after_sec"), 3.0)}')
            ros_args.append(f'-p home_switch_ramp_sec:={_ros_float(request.form.get("home_switch_ramp_sec"), 1.5)}')
        ok, message = procs.start_deploy(policy_pt, ros_args)
        flash(message, 'success' if ok else 'error')
        if not ok:
            # start_deploy() found something already running that this page's stale status check didn't know about
            _suggest_action('Stop policy', url_for('jetson_deploy_toggle', policies_dir=policies_dir,
                                                     task=task, fname=fname, basename=basename))
    return redirect(url_for('jetson_detail', policies_dir=policies_dir, task=task, fname=fname, basename=basename))


@app.route('/jetson/status')
def jetson_status_poll():
    return jsonify({'hw': procs.hardware_bringup_status(), 'deploy': procs.deploy_status()})


@app.route('/jetson/basics/set_home', methods=['POST'])
def jetson_set_home():
    result = ros_actions.set_home()
    flash(result.stdout.strip() or result.stderr.strip() or 'set_home called.',
          'success' if result.ok else 'error')
    return _redirect_next('jetson_home')


@app.route('/jetson/basics/go_to_pose', methods=['POST'])
def jetson_go_to_pose():
    pose_name = request.form.get('pose_name', 'home')
    speed_deg_s = request.form.get('pose_speed_deg_s', '').strip()
    log_csv = bool(request.form.get('log_csv'))
    # Remembered so the field pre-fills with this same value next time instead of always resetting
    procs.set_last_tool_form('go_to_pose', {'speed_deg_s': speed_deg_s, 'log_csv': log_csv})
    result = ros_actions.go_to_pose(pose_name, speed_deg_s=speed_deg_s or None, log_csv=log_csv)
    flash(result.stdout.strip() or result.stderr.strip() or f'go_to_pose {pose_name} called.',
          'success' if result.ok else 'error')
    return _redirect_next('jetson_home')


@app.route('/jetson/basics/set_actuator_params', methods=['POST'])
def jetson_set_actuator_params():
    """Live `ros2 param set /actuator ...` for whichever of position_kp/ position_kd/pose_speed_deg_s were actually filled in."""

    params = {
        name: request.form.get(name, '').strip()
        for name in ('position_kp', 'position_kd', 'pose_speed_deg_s')
    }
    procs.set_last_tool_form('actuator_params', params)
    results = []
    all_ok = True
    for name, value in params.items():
        if not value:
            continue
        result = ros_actions.set_actuator_param(name, value)
        all_ok = all_ok and result.ok
        results.append(f'{name}={value}: '
                        f'{"ok" if result.ok else (result.stderr or result.stdout).strip()[:100]}')
    if not results:
        flash('No values entered -- nothing set.', 'error')
    else:
        flash('; '.join(results), 'success' if all_ok else 'error')
    return _redirect_next('jetson_home')


@app.route('/jetson/basics/read_motor', methods=['POST'])
def jetson_read_motor():
    raw = request.form.get('motor_ids', '').strip()
    motor_ids = [int(x) for x in raw.split(',') if x.strip().isdigit()] if raw else None
    result = ros_actions.read_motor_positions(motor_ids)
    flash(result.stdout.strip() or result.stderr.strip() or 'read_motor_positions called.',
          'success' if result.ok else 'error')
    return _redirect_next('jetson_home')


@app.route('/jetson/basics/adjust_motor', methods=['POST'])
def jetson_adjust_motor():
    motor_id = request.form.get('motor_id')
    degrees = request.form.get('degrees')
    if not motor_id or not degrees:
        flash('motor_id and degrees are both required.', 'error')
        return _redirect_next('jetson_home')
    result = ros_actions.adjust_motor_position(motor_id, degrees)
    flash(result.stdout.strip() or result.stderr.strip() or 'adjust_motor_position called.',
          'success' if result.ok else 'error')
    return _redirect_next('jetson_home')


@app.route('/jetson/basics/reset_motors', methods=['POST'])
def jetson_reset_motors():
    ok, pid = ros_actions.reset_motors()
    flash(f'basic_control restarted (pid {pid}).' if ok else 'Failed to reset motors -- check the Jetson directly.',
          'success' if ok else 'error')
    return _redirect_next('jetson_home')


# ====================================================================== Sheep ======================================================================

@app.route('/sheep')
def sheep_root():
    if procs.is_connected('sheep'):
        return redirect(url_for('sheep_home'))
    return render_template('sheep_connect.html', active_tab='sheep',
                            connect_url=url_for('sheep_connect'), failed=False)


@app.route('/sheep/connect', methods=['POST'])
def sheep_connect():
    ok = ssh.test_connection(SHEEP_HOST)
    procs.set_connected('sheep', ok)
    if ok:
        return redirect(url_for('sheep_home'))
    return render_template('sheep_connect.html', active_tab='sheep',
                            connect_url=url_for('sheep_connect'), failed=True)


def _require_sheep():
    if not procs.is_connected('sheep'):
        return redirect(url_for('sheep_root'))
    return None


@app.route('/sheep/home')
def sheep_home():
    guard = _require_sheep()
    if guard:
        return guard
    folders = [
        {'name': f['name'], 'mtime': f['mtime'], 'url': url_for('sheep_files', folder=f['name'])}
        for f in remote_fs.list_remote_model_folders(SHEEP_HOST)
    ]
    return render_template(
        'folders.html', tab='sheep', active_tab='sheep', sub_tab='models',
        heading='Sheep models/', subtitle=f'{SHEEP_HOST}:{SHEEP_WS_ROOT}/models', folders=folders,
    )


@app.route('/sheep/<folder>')
def sheep_files(folder):
    guard = _require_sheep()
    if guard:
        return guard
    fnames = [
        {**f, 'url': url_for('sheep_checkpoints', folder=folder, fname=f['fname']),
         'download_url': url_for('sheep_download_group', folder=folder, fname=f['fname']),
         'graphs_url': url_for('sheep_trainings_graphs', fname=f['fname'])}
        for f in remote_fs.list_remote_fnames_with_local(SHEEP_HOST, folder)
    ]
    sidebar_items = [
        {'label': f['name'], 'active': f['name'] == folder, 'url': url_for('sheep_files', folder=f['name'])}
        for f in remote_fs.list_remote_model_folders(SHEEP_HOST)
    ]
    return render_template(
        'files.html', tab='sheep', active_tab='sheep', sub_tab='models',
        folder=folder, fnames=fnames, back_url=url_for('sheep_home'), sidebar_items=sidebar_items,
    )


@app.route('/sheep/<folder>/<fname>')
def sheep_checkpoints(folder, fname):
    guard = _require_sheep()
    if guard:
        return guard
    checkpoints = [
        {**c,
         'url': '#',
         'download_url': url_for('sheep_download_one', folder=folder, fname=fname, basename=c['basename']),
         'view_url': url_for('sheep_view_training', folder=folder, fname=fname, basename=c['basename']),
         'delete_url': url_for('sheep_delete_checkpoint', folder=folder, fname=fname, basename=c['basename']),
         'like_url': url_for('sheep_toggle_like', folder=folder, fname=fname, basename=c['basename']),
         'liked': likes.is_liked('sheep', folder, fname, c['basename']),
         'last_start_pose': procs.get_last_tool_form(f"view_training:{c['basename']}").get('start_pose', 'home'),
         'last_max_slew_deg_per_s': procs.get_last_tool_form(f"view_training:{c['basename']}").get('max_slew_deg_per_s', ''),
         'last_joint_stiffness': procs.get_last_tool_form(f"view_training:{c['basename']}").get('joint_stiffness', '')}
        for c in remote_fs.list_remote_checkpoints_with_local(SHEEP_HOST, folder, fname)
    ]
    # back_url: same ?from=graphs marker as local_checkpoints() -- see
    # that route's own comment for the full reasoning.
    back_url = (url_for('sheep_trainings_graphs', fname=fname) if request.args.get('from') == 'graphs'
                else url_for('sheep_files', folder=folder))
    return render_template(
        'checkpoints.html', tab='sheep', active_tab='sheep', sub_tab='models',
        folder=folder, fname=fname, checkpoints=checkpoints,
        env_id=local_fs.resolve_env_id(folder),
        back_url=back_url,
        graphs_url=url_for('sheep_trainings_graphs', fname=fname),
        download_status_url=url_for('sheep_download_status', folder=folder, fname=fname),
        local_folder_url=url_for('local_files', folder=folder),
        tabs_graphs_url=url_for('sheep_trainings_graphs', fname=fname, **{'from': 'checkpoints'}),
        tabs_checkpoints_url=url_for('sheep_checkpoints', folder=folder, fname=fname),
        active_training_tab='checkpoints',
    )


def _sheep_download(folder, basename):
    remote_path = remote_fs.remote_checkpoint_zip_path(folder, basename)
    local_path = local_fs.checkpoint_zip_path(folder, basename)
    result = ssh.scp_download(SHEEP_HOST, remote_path, local_path)
    return result.ok, local_path


@app.route('/sheep/<folder>/<fname>/download', methods=['POST'])
def sheep_download_group(folder, fname):
    ok, message = download_actions.start_group_download(SHEEP_HOST, folder, fname)
    flash(message, 'success' if ok else 'error')
    return redirect(url_for('sheep_checkpoints', folder=folder, fname=fname))


@app.route('/sheep/<folder>/<fname>/download_status')
def sheep_download_status(folder, fname):
    return jsonify(download_actions.get_group_download_status(folder, fname))


@app.route('/sheep/<folder>/<fname>/<basename>/download', methods=['POST'])
def sheep_download_one(folder, fname, basename):
    ok, local_path = _sheep_download(folder, basename)
    flash(f'Downloaded to {local_path}.' if ok else 'Download failed.', 'success' if ok else 'error')
    if ok:
        _suggest_redirect('Go to download', url_for('local_files', folder=folder))
    return redirect(url_for('sheep_checkpoints', folder=folder, fname=fname))


@app.route('/sheep/<folder>/<fname>/<basename>/delete', methods=['POST'])
def sheep_delete_checkpoint(folder, fname, basename):
    guard = _require_sheep()
    if guard:
        return guard
    remote_path = remote_fs.remote_checkpoint_zip_path(folder, basename)
    ok = ssh.remove_remote_file(SHEEP_HOST, remote_path)
    flash(f'Deleted {basename} from sheep.' if ok else f'Failed to delete {basename}.', 'success' if ok else 'error')
    return redirect(url_for('sheep_checkpoints', folder=folder, fname=fname))


@app.route('/sheep/<folder>/<fname>/<basename>/view', methods=['POST'])
def sheep_view_training(folder, fname, basename):
    return _do_view_training('sheep', folder, fname, basename,
                              url_for('sheep_checkpoints', folder=folder, fname=fname))


@app.route('/sheep/<folder>/<fname>/<basename>/like', methods=['POST'])
def sheep_toggle_like(folder, fname, basename):
    likes.toggle_like('sheep', folder, fname, basename)
    return redirect(url_for('sheep_checkpoints', folder=folder, fname=fname))


@app.route('/sheep/trainings')
def sheep_trainings():
    guard = _require_sheep()
    if guard:
        return guard
    running = [
        {**r, 'graphs_url': url_for('sheep_trainings_graphs', fname=r['fname']),
         'stop_url': url_for('sheep_trainings_stop', fname=r['fname']),
         'checkpoint_count': remote_fs.count_remote_checkpoints_for_fname(SHEEP_HOST, r['fname'])}
        for r in remote_fs.list_remote_running_trainings(SHEEP_HOST)
    ]
    model_folders = [f['name'] for f in remote_fs.list_remote_model_folders(SHEEP_HOST)]
    return render_template(
        'trainings.html', tab='sheep', active_tab='sheep', sub_tab='trainings',
        running=running, model_folders=model_folders,
        launch_url=url_for('sheep_trainings_launch'),
        build=build_actions.sheep_build_status(),
        build_url=url_for('sheep_trainings_build'),
        build_status_url=url_for('sheep_trainings_build_status'),
        gpu_status=remote_fs.list_sheep_gpu_status(SHEEP_HOST),
    )


@app.route('/sheep/trainings/launch', methods=['POST'])
def sheep_trainings_launch():
    ok, message = training_actions.launch_remote_training(request.form)
    flash(message, 'success' if ok else 'error')
    return redirect(url_for('sheep_trainings'))


@app.route('/sheep/trainings/build', methods=['POST'])
def sheep_trainings_build():
    guard = _require_sheep()
    if guard:
        return guard
    ok, message = build_actions.start_sheep_build()
    flash(message, 'success' if ok else 'error')
    return redirect(url_for('sheep_trainings'))


@app.route('/sheep/trainings/build_status')
def sheep_trainings_build_status():
    return jsonify(build_actions.sheep_build_status())


@app.route('/sheep/trainings/<fname>/stop', methods=['POST'])
def sheep_trainings_stop(fname):
    ok = training_actions.stop_remote_training(fname)
    flash(f'Stopped "{fname}" on sheep.' if ok else f'Failed to stop "{fname}" on sheep -- check it directly.',
          'success' if ok else 'error')
    return redirect(url_for('sheep_trainings'))


@app.route('/sheep/trainings/<fname>/graphs')
def sheep_trainings_graphs(fname):
    guard = _require_sheep()
    if guard:
        return guard
    scalars = graphs.sheep_sync_and_read_scalars(fname)
    running = any(t['fname'] == fname for t in remote_fs.list_remote_running_trainings(SHEEP_HOST))
    folder = remote_fs.find_remote_folder_for_fname(SHEEP_HOST, fname)
    # back_url: same ?from=checkpoints marker as local_trainings_graphs()
    # -- see that route's own comment.
    back_url = (url_for('sheep_checkpoints', folder=folder, fname=fname)
                if request.args.get('from') == 'checkpoints' and folder
                else url_for('sheep_trainings'))
    return render_template(
        'graphs.html', tab='sheep', active_tab='sheep',
        fname=fname, running=running, sheep=True,
        request_path=request.path, tags=graphs.ordered_tags(scalars), series_json=json.dumps(scalars),
        back_url=back_url,
        tabs_graphs_url=request.path,
        tabs_checkpoints_url=(url_for('sheep_checkpoints', folder=folder, fname=fname, **{'from': 'graphs'})
                               if folder else None),
        active_training_tab='graphs',
    )


# ====================================================================== Liked ====================================================================== Cross-cutting: a liked checkpoint can be 'local' or 'sheep' (see likes.py's own module docstring for why host is part of a like's identity, not just a display label)

@app.route('/liked')
def liked_page():
    items = []
    for item in likes.list_likes():
        host, folder, fname, basename = item['host'], item['folder'], item['fname'], item['basename']
        last_form = procs.get_last_tool_form(f'view_training:{basename}')
        items.append({
            **item,
            'env_id': local_fs.resolve_env_id(folder),
            # Whether this checkpoint's .zip is ALREADY on disk locally
            'downloaded_locally': os.path.exists(local_fs.checkpoint_zip_path(folder, basename)),
            'view_url': url_for('liked_view_training', host=host, folder=folder, fname=fname, basename=basename),
            'unlike_url': url_for('liked_unlike', host=host, folder=folder, fname=fname, basename=basename),
            'checkpoints_url': url_for('local_checkpoints' if host == 'local' else 'sheep_checkpoints',
                                        folder=folder, fname=fname),
            'last_start_pose': last_form.get('start_pose', 'home'),
            'last_max_slew_deg_per_s': last_form.get('max_slew_deg_per_s', ''),
            'last_joint_stiffness': last_form.get('joint_stiffness', ''),
        })
    return render_template('liked.html', tab='liked', active_tab='liked', items=items)


@app.route('/liked/<host>/<folder>/<fname>/<basename>/view', methods=['POST'])
def liked_view_training(host, folder, fname, basename):
    return _do_view_training(host, folder, fname, basename, url_for('liked_page'))


@app.route('/liked/<host>/<folder>/<fname>/<basename>/unlike', methods=['POST'])
def liked_unlike(host, folder, fname, basename):
    likes.toggle_like(host, folder, fname, basename)
    return redirect(url_for('liked_page'))


# ======================================================================

def main():
    port = int(os.environ.get('DASHBOARD_PORT', '5055'))
    threading.Timer(1.0, lambda: webbrowser.open(f'http://127.0.0.1:{port}/')).start()
    app.run(host='127.0.0.1', port=port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
