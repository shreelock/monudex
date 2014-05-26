%% FOR GENERATE MATRICES THAT CONTAIN SURF COEFFICIENTS OF THE TARINING DATASET AND TEST DATABASE
clear;
tic
%TRAINING DATABASE

for i=1:5
    
    [b, path] = fileattrib(sprintf('C:\\Users\\Dutt\\Documents\\MATLAB\\training_dataset\\monu%d\\*onu*',i));
    paths{i}=path;
end
noofPoints=40;
iter=1;
surf_feat=zeros(length(path)*5,noofPoints*64);
Grpvc=zeros(5*length(path),1);

for setnumber=1:5
    pths=paths{setnumber};
    
    for imagenumber=1:length(path)
        tic
        dsc=0;
        image=imread(pths(imagenumber).Name);
        
        %SURF FEATURES
        H = fspecial('unsharp');
        im = imfilter(image,H,'replicate');
        i4surf=rgb2gray(imresize(im,0.35));
        Ipts=OpenSurf(i4surf);
        i_last=min(noofPoints,length(Ipts));
        for lol=1:i_last
            k=Ipts(lol).descriptor;
            dsc=[dsc k'];
        end
          dsc=dsc(2:end);
          dsc(length(dsc):noofPoints*64)=0;
        surf_feat(iter,:)=dsc;
        %FORMALITY
        toc
        disp(iter);
        Grpvc(iter)=setnumber;
        iter=iter+1;
    end
end

fprintf('Trained .. Paused .. Enter \n');
pause;

%% SURF DATA FOR TEST IMAGES

    
    [b, path] = fileattrib('C:\Users\Dutt\Documents\MATLAB\test_dataset_9\monu1\Monu*');
    paths=path;

iter=1;
setnumber=1;
par=1:20;
jar=1:20;
tsurf_feat=zeros(length(jar),noofPoints*64);
checkMat=zeros(1,length(jar));

    pths=paths;
    
    for imagenumber=1:20
        tic
        dsc=0;
        
        image=imread(path(imagenumber).Name);
        %for generating the image names besides the answer matrix.
        %-----------------
        s=path(imagenumber).Name;
        k=strfind(s,'\monu1\Monu');
        j=k+11;
        k1=strfind(s,'.');
        k1=k1-1;
        ss=s(j:k1);
        p=str2num(ss);
        %-----------------
        %SURF FEATURES
        H = fspecial('unsharp');
        im = imfilter(image,H,'replicate');
        i4surf=rgb2gray(imresize(im,0.35));
        Ipts=OpenSurf(i4surf);
        for lol=1:noofPoints
            k=Ipts(lol).descriptor;
            dsc=[dsc k'];
        end
          dsc=dsc(2:end);
        tsurf_feat(iter,:)=dsc;
        %FORMALITY
        toc
        disp(iter); 
        checkMat(iter)=p;
        iter=iter+1;
    end
toc    
fprintf('Ready for Test .. Paused .. Enter \n');

pause;