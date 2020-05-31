def get_stages_and_cfgs(fusion_cfg, stages, params):
    layer_num = fusion_cfg.layer_num
    bn_relu = fusion_cfg.get_bn_relu()

    stage_dict = {}
    layer_output_dict = {} # A dict for the synonym of the output of each layer
    param_dict = {}
    inputs_cfg = {}
    filters_cfg = {}
    outputs_cfg = {}
    stage_pt = 1 # Skip input
    param_pt = 1 # Skip input

    for l in range(0, layer_num):
        if fusion_cfg.need_padding(l):
            stage_dict['PaddedInput_{}'.format(l)] = stages[stage_pt][0]
            stage_pt += 1
        if bn_relu[l]:
            stage_dict['Output_{}'.format(l)], \
                stage_dict['Output_{}_ScaleShift'.format(l)], \
                    stage_dict['Output_{}_ReLU'.format(l)] = stages[stage_pt]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}_ReLU'.format(l)]
            param_dict['Filter_{}'.format(l)], \
                param_dict['Scale_{}'.format(l)], \
                    param_dict['Shift_{}'.format(l)] = params[param_pt]
        else:
            stage_dict['Output_{}'.format(l)] = stages[stage_pt][0]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}'.format(l)]
            param_dict['Filter_{}'.format(l)] = params[param_pt][0]

        inputs_cfg['Layer_{}'.format(l)] = fusion_cfg.get_input(l)
        filters_cfg['Layer_{}'.format(l)] = fusion_cfg.get_filter(l)
        outputs_cfg['Layer_{}'.format(l)] = fusion_cfg.get_output(l)

        stage_pt += 1
        param_pt += 1

    return stage_dict, layer_output_dict, param_dict, inputs_cfg, filters_cfg, outputs_cfg

