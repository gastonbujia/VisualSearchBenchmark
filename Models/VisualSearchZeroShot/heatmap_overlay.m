
% img = image on which to overlay heatmap
% heatmap = the heatmap
% (optional) colorfunc .. this can be 'jet' , or 'hot' , or 'flag'

function omap = heatmap_overlay( img , heatmap, colorfun )

if ( strcmp(class(img),'char') == 1 ) img = imread(img); end
if ( strcmp(class(img),'uint8') == 1 ) img = double(img)/256; end

szh = size(heatmap);
szi = size(img);

if ( (szh(1)~=szi(1)) | (szh(2)~=szi(2)) )
  heatmap = imresize( heatmap , [ szi(1) szi(2) ] , 'bicubic' );
end
  
if ( size(img,3) == 1 )
  img = repmat(img,[1 1 3]);
end
  
if ( nargin == 2 )
    colorfun = 'jet';
end
colorfunc = eval(sprintf('%s(50)',colorfun));

%img(find(img>=1))=0.999;
%heatmap(find(heatmap>=1))=0.999;

heatmap = double(heatmap) / max(heatmap(:));
omap1 = 0.8*(1-repmat(heatmap.^0.8,[1 1 3])).*double(img)/max(double(img(:)));
omap2 = repmat(heatmap.^0.8,[1 1 3]).* shiftdim(reshape( interp2(1:3,1:50,colorfunc,1:3,1+49*reshape( heatmap , [ prod(size(heatmap))  1 ] ))',[ 3 size(heatmap) ]),1);
omap =  omap1+omap2; 
omap = real(omap);
%omap = 0.7*(1-repmat(heatmap.^0.7,[1 1 3])).*double(img)/max(double(img(:))) + repmat(heatmap.^0.7,[1 1 3]).* shiftdim(reshape( interp2(1:3,1:50,colorfunc,1:3,1+49*reshape( heatmap , [ prod(size(heatmap))  1 ] ))',[ 3 size(heatmap) ]),1);
%omap = real(omap);