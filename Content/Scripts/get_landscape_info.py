import unreal_engine as ue
from unreal_engine.classes import Landscape


def main():
    """Create 3 spline components (L,R,Center) from landscape spline components"""
    world = ue.get_editor_world()
    objects = world.all_objects()
    lspline_ctrl_pts = []
    for o in objects:
        try:
            if o.get_full_name().startswith('LandscapeSplineControlPoint '):
                lspline_ctrl_pts.append(o)
        except:
            print('error getting object')

    # for o in lspline_ctrl_pts:
    #     print(o.get_full_name())

    for pt in lspline_ctrl_pts:
        # print(dir(pt))
        # print(pt.get_points())
        print('point location ' + str(pt.Location))
        print(pt.SideFalloff)
        print(pt.SegmentMeshOffset)
        print(len(pt.Points))
        if len(pt.Points) > 0:
            print('point center ' + str(pt.Points[0].Center))
        # print('length of interp points %d' % len(pt.GetPoints()))

    print('num spline ctrl points ' + str(len(lspline_ctrl_pts)))



if __name__ == '__main__':
    main()


"""
LogPython: ['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'actor_begin_play', 'actor_create_default_subobject', 
'actor_destroy', 'actor_destroy_component', 'actor_has_component_of_type', 'actor_has_tag', 'actor_set_level_sequence', 'actor_spawn', 'add_actor_component', 'add_actor_local_offset', 'add_actor_local_rotation', 'add_actor_root_component', 'add_actor_world_offset', 'add_actor_world_rotation', 'add_angular_impulse', 'add_anim_composite_section', 'add_controll
er_pitch_input', 'add_controller_roll_input', 'add_controller_yaw_input', 'add_foliage_asset', 'add_force', 'add_function', 'add_impulse', 'add_instance_component', 'add_key_to_sequence', 'add_movement_input', 'add_new_raw_track', 'add_property', 'add_property_flags', 'add_python_component', 'add_to_root', 'add_torque', 'add_viewport_widget_content', 'all_ac
tors', 'all_objects', 'anim_get_skeleton', 'anim_set_skeleton', 'apply_raw_anim_changes', 'as_dict', 'asset_can_reimport', 'asset_import_data', 'asset_import_data_set_sources', 'asset_reimport', 'attach_to_actor', 'attach_to_component', 'auto_root', 'bind_action', 'bind_axis', 'bind_event', 'bind_input_axis', 'bind_key', 'bind_pressed_key', 'bind_released_ke
y', 'broadcast', 'call', 'call_function', 'can_crouch', 'can_jump', 'can_modify', 'capture_initialize', 'capture_load_from_config', 'capture_start', 'capture_stop', 'class_generated_by', 'class_get_config_name', 'class_get_flags', 'class_set_config_name', 'class_set_flags', 'clear_obj_flags', 'component_has_tag', 'component_is_registered', 'component_type_re
gistry_invalidate_class', 'components', 'conditional_begin_destroy', 'create_default_subobject', 'create_landscape_info', 'create_material_instance_dynamic', 'create_player', 'create_widget', 'crouch', 'data_table_add_row', 'data_table_as_dict', 'data_table_as_json', 'data_table_find_row', 'data_table_get_all_rows', 'data_table_remove_row', 'data_table_renam
e_row', 'delegate_bind_ufunction', 'destroy_actor_component', 'destroy_component', 'destructible_apply_damage', 'disown', 'draw_debug_line', 'duplicate', 'enable_click_events', 'enable_input', 'enable_mouse_over_events', 'enum_names', 'enum_user_defined_names', 'enum_values', 'export_to_file', 'extract_bone_transform', 'extract_root_motion', 'factory_create_
new', 'factory_import_object', 'find_actor_by_label', 'find_function', 'find_object', 'from_bytes', 'functions', 'game_viewport_client_get_window', 'game_viewport_client_set_rendering_flag', 'get_actor_bounds', 'get_actor_component', 'get_actor_component_by_class', 'get_actor_component_by_type', 'get_actor_components', 'get_actor_components_by_class', 'get_a
ctor_components_by_tag', 'get_actor_components_by_type', 'get_actor_forward', 'get_actor_label', 'get_actor_location', 'get_actor_right', 'get_actor_root_component', 'get_actor_rotation', 'get_actor_scale', 'get_actor_transform', 'get_actor_up', 'get_actor_velocity', 'get_all_child_actors', 'get_anim_instance', 'get_archetype', 'get_archetype_instances', 'ge
t_attached_actors', 'get_available_audio_byte_count', 'get_blend_parameter', 'get_bone_transform', 'get_cdo', 'get_class', 'get_component', 'get_component_by_class', 'get_component_by_type', 'get_components', 'get_components_by_class', 'get_components_by_tag', 'get_components_by_type', 'get_control_rotation', 'get_controlled_pawn', 'get_controller', 'get_cur
rent_level', 'get_display_name', 'get_editor_world_counterpart_actor', 'get_folder_path', 'get_foliage_instances', 'get_foliage_types', 'get_forward_vector', 'get_full_name', 'get_game_viewport', 'get_hit_result_under_cursor', 'get_hud', 'get_inner', 'get_input_axis', 'get_instanced_foliage_actor_for_current_level', 'get_instanced_foliage_actor_for_level', '
get_key_prop', 'get_landscape_info', 'get_levels', 'get_material_graph', 'get_material_scalar_parameter', 'get_material_static_switch_parameter', 'get_material_texture_parameter', 'get_material_vector_parameter', 'get_metadata', 'get_name', 'get_num_players', 'get_num_spectators', 'get_obj_flags', 'get_outer', 'get_outermost', 'get_overlapping_actors', 'get_
owner', 'get_path_name', 'get_pawn', 'get_physics_angular_velocity', 'get_physics_linear_velocity', 'get_player_camera_manager', 'get_player_controller', 'get_player_hud', 'get_player_pawn', 'get_property', 'get_property_array_dim', 'get_property_class', 'get_property_flags', 'get_property_struct', 'get_py_proxy', 'get_raw_animation_data', 'get_raw_animation
_track', 'get_raw_mesh', 'get_relative_location', 'get_relative_rotation', 'get_relative_scale', 'get_relative_transform', 'get_right_vector', 'get_socket_actor_transform', 'get_socket_location', 'get_socket_rotation', 'get_socket_transform', 'get_socket_world_transform', 'get_spline_length', 'get_super_class', 'get_thumbnail', 'get_up_vector', 'get_upropert
y', 'get_value_prop', 'get_world', 'get_world_delta_seconds', 'get_world_location', 'get_world_location_at_distance_along_spline', 'get_world_rotation', 'get_world_scale', 'get_world_transform', 'get_world_type', 'graph_add_node', 'graph_add_node_call_function', 'graph_add_node_custom_event', 'graph_add_node_dynamic_cast', 'graph_add_node_event', 'graph_add_
node_variable_get', 'graph_add_node_variable_set', 'graph_get_good_place_for_new_node', 'has_metadata', 'has_property', 'has_world', 'hud_draw_2d_line', 'hud_draw_line', 'hud_draw_rect', 'hud_draw_text', 'hud_draw_texture', 'import_custom_properties', 'input_axis', 'input_key', 'is_a', 'is_action_pressed', 'is_action_released', 'is_child_of', 'is_crouched', 
'is_falling', 'is_flying', 'is_input_key_down', 'is_jumping', 'is_owned', 'is_rooted', 'is_valid', 'jump', 'landscape_export_to_raw_mesh', 'landscape_import', 'launch', 'line_trace_multi_by_channel', 'line_trace_single_by_channel', 'make_unique_object_name', 'modify', 'morph_target_get_deltas', 'morph_target_populate_deltas', 'node_allocate_default_pins', 'n
ode_create_pin', 'node_find_pin', 'node_function_entry_set_pure', 'node_get_title', 'node_pin_default_value_changed', 'node_pin_type_changed', 'node_pins', 'node_reconstruct', 'own', 'package_get_filename', 'package_is_dirty', 'play', 'play_sound_at_location', 'posses', 'post_edit_change', 'post_edit_change_property', 'pre_edit_change', 'project_world_locati
on_to_screen', 'properties', 'queue_audio', 'quit_game', 'register_component', 'remove_all_viewport_widgets', 'remove_from_root', 'remove_viewport_widget_content', 'render_target_get_data', 'render_target_get_data_to_buffer', 'render_thumbnail', 'reset_audio', 'reset_obj_flags', 'restart_level', 'save_config', 'save_package', 'sequencer_add_actor', 'sequence
r_add_actor_component', 'sequencer_add_camera_cut_track', 'sequencer_add_master_track', 'sequencer_add_possessable', 'sequencer_add_track', 'sequencer_changed', 'sequencer_create_folder', 'sequencer_find_possessable', 'sequencer_find_spawnable', 'sequencer_folders', 'sequencer_get_camera_cut_track', 'sequencer_get_display_name', 'sequencer_import_fbx_transfo
rm', 'sequencer_make_new_spawnable', 'sequencer_master_tracks', 'sequencer_possessable_tracks', 'sequencer_possessables', 'sequencer_possessables_guid', 'sequencer_remove_camera_cut_track', 'sequencer_remove_master_track', 'sequencer_remove_possessable', 'sequencer_remove_spawnable', 'sequencer_remove_track', 'sequencer_section_add_key', 'sequencer_sections'
, 'sequencer_set_display_name', 'sequencer_set_playback_range', 'sequencer_set_section_range', 'sequencer_track_add_section', 'sequencer_track_sections', 'set_actor_hidden_in_game', 'set_actor_label', 'set_actor_location', 'set_actor_rotation', 'set_actor_scale', 'set_actor_transform', 'set_blend_parameter', 'set_current_level', 'set_folder_path', 'set_level
_sequence_asset', 'set_material', 'set_material_by_name', 'set_material_parent', 'set_material_scalar_parameter', 'set_material_static_switch_parameter', 'set_material_texture_parameter', 'set_material_vector_parameter', 'set_metadata', 'set_name', 'set_obj_flags', 'set_outer', 'set_physics_angular_velocity', 'set_physics_linear_velocity', 'set_player_hud', 
'set_property', 'set_property_flags', 'set_relative_location', 'set_relative_rotation', 'set_relative_scale', 'set_relative_transform', 'set_simulate_physics', 'set_skeletal_mesh', 'set_slate_widget', 'set_timer', 'set_view_target', 'set_world_location', 'set_world_rotation', 'set_world_scale', 'set_world_transform', 'show_mouse_cursor', 'simple_move_to_loca
tion', 'skeletal_mesh_build_lod', 'skeletal_mesh_get_active_bone_indices', 'skeletal_mesh_get_bone_map', 'skeletal_mesh_get_lod', 'skeletal_mesh_get_raw_indices', 'skeletal_mesh_get_required_bones', 'skeletal_mesh_get_soft_vertices', 'skeletal_mesh_lods_num', 'skeletal_mesh_register_morph_target', 'skeletal_mesh_sections_num', 'skeletal_mesh_set_active_bone_
indices', 'skeletal_mesh_set_bone_map', 'skeletal_mesh_set_required_bones', 'skeletal_mesh_set_skeleton', 'skeletal_mesh_set_soft_vertices', 'skeletal_mesh_to_import_vertex_map', 'skeleton_add_bone', 'skeleton_bones_get_num', 'skeleton_find_bone_index', 'skeleton_get_bone_name', 'skeleton_get_parent_index', 'skeleton_get_ref_bone_pose', 'sound_get_data', 'so
und_set_data', 'static_mesh_build', 'static_mesh_create_body_setup', 'static_mesh_generate_kdop10x', 'static_mesh_generate_kdop10y', 'static_mesh_generate_kdop10z', 'static_mesh_generate_kdop18', 'static_mesh_generate_kdop26', 'static_mesh_set_collision_for_lod', 'static_mesh_set_shadow_for_lod', 'stop_jumping', 'struct_add_variable', 'struct_get_variables',
 'struct_move_variable_down', 'struct_move_variable_up', 'struct_remove_variable', 'take_widget', 'texture_get_data', 'texture_get_height', 'texture_get_source_data', 'texture_get_width', 'texture_set_data', 'texture_set_source_data', 'texture_update_resource', 'to_bytearray', 'to_bytes', 'uncrouch', 'unposses', 'unregister_component', 'update_compressed_tra
ck_map_from_raw', 'update_raw_track', 'vlog', 'vlog_cylinder', 'was_input_key_just_pressed', 'was_input_key_just_released', 'world_create_folder', 'world_delete_folder', 'world_exec', 'world_folders', 'world_rename_folder', 'world_tick']
"""