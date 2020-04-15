#include "../darknet/include/darknet.h"

typedef struct {
    int h, w, c;
} shape;


shape get_net_output_shape(network* net)
{
    layer output = get_network_output_layer(net);
    shape s;
    s.h = output.out_h;
    s.w = output.out_w;
    s.c = output.out_c;
    return s;
}


int has_input(layer* l, int n) {
    for (int i=0; i<l->n; i++) {
        if (l->input_layers[i] == n) {
            return 1;
        }
    }

    return 0;
}

typedef struct {
    int h, w, c;
    float* output;
} layer_output;

typedef struct {
    int count;
    layer_output* output;

} net_outputs;


net_outputs get_net_outputs(network* net)
{
    int indices[net->n];
    int count = 0;

    int i = net->n-1;

    for(; i >= 0; --i) {
        // this is as in darknet's get_network_output_layer
        if (net->layers[i].type != COST) {
            indices[count++] = i;
            break;
        }
    }

    for(; i > 0; --i) {
        if (net->layers[i].type == ROUTE) {
            // if previous layer isn't an input to the route, it's a network output (leaf on tree)
            if (!has_input(&net->layers[i], i-1)) {
                indices[count++] = i-1;
            }
        }
    }

    net_outputs outputs;
    outputs.count = count;
    outputs.output = calloc(count, sizeof(layer_output));

    for (i=0; i<count; ++i) {
        layer l = net->layers[indices[count-i-1]];

        outputs.output[i].h = l.out_h;
        outputs.output[i].w = l.out_w;
        outputs.output[i].c = l.out_c;
        outputs.output[i].output = l.output;
    }

    return outputs;
}

void free_net_outputs(net_outputs netout) {
    free(netout.output);
}
