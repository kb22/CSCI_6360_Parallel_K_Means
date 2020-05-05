clc
clear all
close all

%Original Image
A = double(imread('x3.jpg'));
A = A / 255; % Divide by 255 so that all values are in the range 0 - 1
img_size = size(A);

%Image from C program
fileID=fopen('output.bin');
X_bin=fread(fileID,'double');
fclose(fileID);
X1=reshape(X_bin', 3, img_size(1)*img_size(2)).';
A2 = reshape(X1, img_size(1), img_size(2), 3);

% outputImage = imresize(A2, [img_size(1) img_size(2)]);
% imwrite(outputImage , 'compressed1.jpg')

%Display original image
subplot(1, 2, 1);
imagesc(A);
title('Original');

% Display compressed image
subplot(1, 2, 2);
imagesc(A2)
title(sprintf('Compressed image'));
