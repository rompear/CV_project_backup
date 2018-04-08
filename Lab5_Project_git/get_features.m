function [frames, des] = get_features(image, sift_type, color_space, step_size)
    % calculate features
    [frames, des] = vl_sift(single(rgb2gray(image)));

    % get channels
    if color_space == "rgb"
        image = double(image);
        channel = get_rgb_channels(image);
    elseif color_space == "RGB"
        image = double(image);
        channel = get_RGB_channels(image);
    elseif color_space == "opponent"
        image = double(image);
        channel = get_opponent_channels(image);
    elseif color_space == "l"
        image = double(image);
        channel = get_l_channels(image);

    elseif color_space == "gray"
        if sift_type == "dense"
            [frames, des] = vl_phow(single(rgb2gray(image)), 'step', step_size);
        end
        
        return
    end
    
    % get sift type
    if sift_type == "dense"
        [~, d1] = vl_phow(single(channel(:,:,1)), 'step', step_size);
        [~, d2] = vl_phow(single(channel(:,:,2)), 'step', step_size);
        [~, d3] = vl_phow(single(channel(:,:,3)), 'step', step_size);
        des = vertcat(d1, d2, d3);
    else
        [~, d1] = vl_sift(single(channel(:,:,1)), 'frames', frames);
        [~, d2] = vl_sift(single(channel(:,:,2)), 'frames', frames);
        [~, d3] = vl_sift(single(channel(:,:,3)), 'frames', frames);
        des = vertcat(d1, d2, d3);
    end
end

function channel = get_rgb_channels(image)
    channel(:,:,1) = 255 .* (image(:,:,1) ./ (image(:,:,1) + image(:,:,2) + image(:,:,3)));
    channel(:,:,2) = 255 .* (image(:,:,2) ./ (image(:,:,1) + image(:,:,2) + image(:,:,3)));
    channel(:,:,3) = 255 .* (image(:,:,3) ./ (image(:,:,1) + image(:,:,2) + image(:,:,3)));
end

function channel = get_RGB_channels(image)
    channel(:,:,1) = image(:,:,1);
    channel(:,:,2) = image(:,:,2);
    channel(:,:,3) = image(:,:,3);
end

function channel = get_opponent_channels(image)
    channel(:,:,1) = (image(:,:,1) - image(:,:,2)) ./ sqrt(2);
    channel(:,:,2) = (image(:,:,1) + image(:,:,2) - 2 .* image(:,:,3)) ./ sqrt(6);
    channel(:,:,3) = (image(:,:,1) + image(:,:,2) + image(:,:,3)) ./ sqrt(2);
end

function channel = get_l_channels(image)
    total = (image(:,:,1) - image(:,:,2)) .^ 2 + (image(:,:,1) - image(:,:,3)) .^ 2 + (image(:,:,2) - image(:,:,3)) .^ 2;
    channel(:,:,1) = (image(:,:,1) - image(:,:,2)) .^ 2 ./ total;
    channel(:,:,2) = (image(:,:,1) - image(:,:,3)) .^ 2 ./ total;
    channel(:,:,3) = (image(:,:,2) - image(:,:,3)) .^ 2 ./ total;
end


