try
    warning('on','MATLAB:project:api:NoProjectCurrentlyLoaded')
    proj = currentProject;
    close(proj);
    clear proj
catch ME
    warning(ME.message);
end